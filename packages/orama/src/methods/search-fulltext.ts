import { getFacets } from '../components/facets.js'
import { getGroups } from '../components/groups.js'
import { runAfterSearch, runBeforeSearch } from '../components/hooks.js'
import { getInternalDocumentId } from '../components/internal-document-id-store.js'
import { searchByGeoWhereClause } from '../components/index.js'
import { applyPinningRules } from '../components/pinning-manager.js'
import { Language } from '../components/tokenizer/languages.js'
import { createError } from '../errors.js'
import type {
  AnyOrama,
  BM25Params,
  CustomSorterFunctionItem,
  ElapsedTime,
  Results,
  SearchParamsFullText,
  TokenScore,
  TypedDocument
} from '../types.js'
import { getNanosecondsTime, removeVectorsFromHits, sortTokenScorePredicate } from '../utils.js'
import { count } from './docs.js'
import { fetchDocuments, fetchDocumentsWithDistinct } from './search.js'

/**
 * Filters search results to only include documents that match exactly when exact mode is enabled.
 * 
 * @param orama - The Orama instance
 * @param results - The search results to filter
 * @param term - The original search term (may include leading/trailing spaces)
 * @param propertiesToSearch - The properties to search in
 * @param preserveSpaces - Whether to enforce exact matching of leading/trailing spaces
 * @returns Filtered results containing only exact matches
 */
function filterExactMatches<T extends AnyOrama>(
  orama: T,
  results: TokenScore[],
  term: string,
  propertiesToSearch: string[],
  preserveSpaces: boolean
): TokenScore[] {
  const trimmedTerm = term.trim()
  const hasLeadingSpace = term.startsWith(' ')
  const hasTrailingSpace = term.endsWith(' ')
  
  // Handle empty/whitespace-only terms
  if (!trimmedTerm) {
    if (preserveSpaces && (hasLeadingSpace || hasTrailingSpace)) {
      return filterBySpaceMatch(results, orama, term, propertiesToSearch)
    }
    return results // Empty term matches all
  }
  
  const searchTerms = trimmedTerm.split(/\s+/).filter(Boolean)
  if (searchTerms.length === 0) {
    return results
  }
  
  // Note: Document retrieval is done per-document in the filter loop.
  // For large result sets, consider batch retrieval optimization if performance becomes an issue.
  return results.filter(([docId]) => {
    const doc = orama.documentsStore.get(orama.data.docs, docId)
    if (!doc) return false
    
    return propertiesToSearch.some((prop) =>
      matchesExactTerm(doc, prop, term, searchTerms, preserveSpaces, hasLeadingSpace, hasTrailingSpace)
    )
  })
}

/**
 * Filters results for whitespace-only terms with preserveSpaces enabled.
 */
function filterBySpaceMatch<T extends AnyOrama>(
  results: TokenScore[],
  orama: T,
  term: string,
  propertiesToSearch: string[]
): TokenScore[] {
  return results.filter(([docId]) => {
    const doc = orama.documentsStore.get(orama.data.docs, docId)
    if (!doc) return false
    
    return propertiesToSearch.some((prop) => {
      const propValue = getPropValue(doc, prop)
      return typeof propValue === 'string' && propValue.includes(term)
    })
  })
}

/**
 * Checks if a document property matches the exact search term.
 * 
 * @param doc - The document to check
 * @param prop - The property path to check
 * @param originalTerm - The original search term (with spaces)
 * @param searchTerms - The split search terms (without spaces)
 * @param preserveSpaces - Whether strict space matching is enabled
 * @param hasLeadingSpace - Whether the original term has leading space
 * @param hasTrailingSpace - Whether the original term has trailing space
 * @returns True if the property matches the exact term
 */
function matchesExactTerm(
  doc: any,
  prop: string,
  originalTerm: string,
  searchTerms: string[],
  preserveSpaces: boolean,
  hasLeadingSpace: boolean,
  hasTrailingSpace: boolean
): boolean {
  const propValue = getPropValue(doc, prop)
  if (typeof propValue !== 'string') return false
  
  // Handle strict space matching
  if (preserveSpaces && (hasLeadingSpace || hasTrailingSpace)) {
    if (!propValue.includes(originalTerm)) {
      return false
    }
  }
  
  // Check if all search terms appear as complete words in the property value
  // Note: Case-sensitive matching is intentional for exact mode, even though
  // Orama is case-insensitive by design. This provides the exact matching behavior
  // users expect when exact: true is specified.
  return searchTerms.every((searchTerm) => {
    // Check if the term contains only word characters (ASCII word characters only)
    // Note: \w matches [A-Za-z0-9_], not Unicode letters. This is intentional
    // for consistency with JavaScript regex word boundaries.
    const isWordOnly = /^\w+$/.test(searchTerm)
    
    if (isWordOnly) {
      // For word-only terms, use word boundaries to prevent partial matches
      const regex = new RegExp(`\\b${escapeRegex(searchTerm)}\\b`)
      return regex.test(propValue)
    } else {
      // For terms with special characters, match directly (they're already exact)
      const escaped = escapeRegex(searchTerm)
      return new RegExp(escaped).test(propValue)
    }
  })
}

/**
 * Escapes special regex characters in a string to make it safe for use in a RegExp.
 * 
 * @param str - The string to escape
 * @returns The escaped string safe for use in regex patterns
 * 
 * @example
 * escapeRegex('test.123') // Returns 'test\\.123'
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Retrieves a nested property value from an object using dot notation.
 * 
 * @param obj - The object to traverse
 * @param path - The dot-notation path to the property (e.g., 'user.name')
 * @returns The property value, or undefined if the path doesn't exist
 * 
 * @example
 * getPropValue({ user: { name: 'John' } }, 'user.name') // Returns 'John'
 * getPropValue({ user: { name: 'John' } }, 'user.age') // Returns undefined
 */
function getPropValue(obj: any, path: string): any {
  const keys = path.split('.')
  let value = obj
  for (const key of keys) {
    // Explicit null check: typeof null === 'object' in JavaScript, but we can't access properties on null
    if (value && typeof value === 'object' && value !== null && key in value) {
      value = value[key]
    } else {
      return undefined
    }
  }
  return value
}

export function innerFullTextSearch<T extends AnyOrama>(
  orama: T,
  params: Pick<
    SearchParamsFullText<T>,
    'term' | 'properties' | 'where' | 'exact' | 'preserveSpaces' | 'tolerance' | 'boost' | 'relevance' | 'threshold'
  >,
  language: Language | undefined
) {
  const { term, properties } = params
  
  // Normalize preserveSpaces: it's only effective when exact is true
  // If exact is false, preserveSpaces is automatically disabled
  const preserveSpaces = params.exact ? (params.preserveSpaces ?? false) : false

  const index = orama.data.index
  // Get searchable string properties
  let propertiesToSearch = orama.caches['propertiesToSearch'] as string[]
  if (!propertiesToSearch) {
    const propertiesToSearchWithTypes = orama.index.getSearchablePropertiesWithTypes(index)

    propertiesToSearch = orama.index.getSearchableProperties(index)
    propertiesToSearch = propertiesToSearch.filter((prop: string) =>
      propertiesToSearchWithTypes[prop].startsWith('string')
    )

    orama.caches['propertiesToSearch'] = propertiesToSearch
  }

  if (properties && properties !== '*') {
    for (const prop of properties) {
      if (!propertiesToSearch.includes(prop as string)) {
        throw createError('UNKNOWN_INDEX', prop as string, propertiesToSearch.join(', '))
      }
    }

    propertiesToSearch = propertiesToSearch.filter((prop: string) => (properties as string[]).includes(prop))
  }

  // If filters are enabled, we need to get the IDs of the documents that match the filters.
  const hasFilters = Object.keys(params.where ?? {}).length > 0
  let whereFiltersIDs: Set<number> | undefined
  if (hasFilters) {
    whereFiltersIDs = orama.index.searchByWhereClause(index, orama.tokenizer, params.where!, language)
  }

  let uniqueDocsIDs: TokenScore[]
  // We need to perform the search if:
  // - we have a search term
  // - or we have properties to search
  //   in this case, we need to return all the documents that contains at least one of the given properties
  const threshold = params.threshold !== undefined && params.threshold !== null ? params.threshold : 1

  if (term || properties) {
    const docsCount = count(orama)
    uniqueDocsIDs = orama.index.search(
      index,
      term || '',
      orama.tokenizer,
      language,
      propertiesToSearch,
      params.exact || false,
      params.tolerance || 0,
      params.boost || {},
      applyDefault(params.relevance),
      docsCount,
      whereFiltersIDs,
      threshold
    )

    // When exact is true and we have a term, filter results to only include documents
    // where the original text contains the exact search term (case-sensitive).
    // This is a highly requested feature and although Orama is not case-sensitive by design,
    // this is a reasonable compromise.
    // Note: preserveSpaces is only effective when exact is true. If exact is false, preserveSpaces is ignored.
    if (params.exact && term) {
      uniqueDocsIDs = filterExactMatches(
        orama,
        uniqueDocsIDs,
        term,
        propertiesToSearch,
        preserveSpaces
      )
    }
  } else {
    // Check if this is a geosearch-only query first
    if (hasFilters) {
      const geoResults = searchByGeoWhereClause(index, params.where!)
      if (geoResults) {
        // This is a geosearch-only query with distance scoring
        uniqueDocsIDs = geoResults
      } else {
        // Regular filter query without search term
        const docIds = whereFiltersIDs ? Array.from(whereFiltersIDs) : []
        uniqueDocsIDs = docIds.map((k) => [+k, 0] as TokenScore)
      }
    } else {
      // No search term and no filters - return all documents
      const docIds = Object.keys(orama.documentsStore.getAll(orama.data.docs))
      uniqueDocsIDs = docIds.map((k) => [+k, 0] as TokenScore)
    }
  }

  return uniqueDocsIDs
}

export function fullTextSearch<T extends AnyOrama, ResultDocument = TypedDocument<T>>(
  orama: T,
  params: SearchParamsFullText<T, ResultDocument>,
  language?: string
): Results<ResultDocument> | Promise<Results<ResultDocument>> {
  const timeStart = getNanosecondsTime()

  function performSearchLogic(): Results<ResultDocument> {
    const vectorProperties = Object.keys(orama.data.index.vectorIndexes)
    const shouldCalculateFacets = params.facets && Object.keys(params.facets).length > 0
    const { limit = 10, offset = 0, distinctOn, includeVectors = false } = params
    const isPreflight = params.preflight === true

    let uniqueDocsArray = innerFullTextSearch(orama, params, language)

    if (params.sortBy) {
      if (typeof params.sortBy === 'function') {
        const ids = uniqueDocsArray.map(([id]) => id)
        const docs = orama.documentsStore.getMultiple(orama.data.docs, ids)
        const docsWithIdAndScore: CustomSorterFunctionItem<ResultDocument>[] = docs.map((d, i) => [
          uniqueDocsArray[i][0],
          uniqueDocsArray[i][1],
          d!
        ])
        docsWithIdAndScore.sort(params.sortBy)
        uniqueDocsArray = docsWithIdAndScore.map(([id, score]) => [id, score])
      } else {
        uniqueDocsArray = orama.sorter
          .sortBy(orama.data.sorting, uniqueDocsArray, params.sortBy)
          .map(([id, score]) => [getInternalDocumentId(orama.internalDocumentIDStore, id), score])
      }
    } else {
      uniqueDocsArray = uniqueDocsArray.sort(sortTokenScorePredicate)
    }

    // Apply pinning rules after sorting but before pagination
    uniqueDocsArray = applyPinningRules(orama, orama.data.pinning, uniqueDocsArray, params.term)

    let results
    if (!isPreflight) {
      results = distinctOn
        ? fetchDocumentsWithDistinct(orama, uniqueDocsArray, offset, limit, distinctOn)
        : fetchDocuments(orama, uniqueDocsArray, offset, limit)
    }

    const searchResult: Results<ResultDocument> = {
      elapsed: {
        formatted: '',
        raw: 0
      },
      hits: [],
      count: uniqueDocsArray.length
    }

    if (typeof results !== 'undefined') {
      searchResult.hits = results.filter(Boolean)
      if (!includeVectors) {
        removeVectorsFromHits(searchResult, vectorProperties)
      }
    }

    if (shouldCalculateFacets) {
      const facets = getFacets(orama, uniqueDocsArray, params.facets!)
      searchResult.facets = facets
    }

    if (params.groupBy) {
      searchResult.groups = getGroups<T, ResultDocument>(orama, uniqueDocsArray, params.groupBy)
    }

    searchResult.elapsed = orama.formatElapsedTime(getNanosecondsTime() - timeStart) as ElapsedTime

    return searchResult
  }

  async function executeSearchAsync() {
    if (orama.beforeSearch) {
      await runBeforeSearch(orama.beforeSearch, orama, params, language)
    }

    const searchResult = performSearchLogic()

    if (orama.afterSearch) {
      await runAfterSearch(orama.afterSearch, orama, params, language, searchResult)
    }

    return searchResult
  }

  const asyncNeeded = orama.beforeSearch?.length || orama.afterSearch?.length
  if (asyncNeeded) {
    return executeSearchAsync()
  }

  return performSearchLogic()
}

export const defaultBM25Params: BM25Params = {
  k: 1.2,
  b: 0.75,
  d: 0.5
}
function applyDefault(bm25Relevance?: BM25Params): Required<BM25Params> {
  const r = bm25Relevance ?? {}
  r.k = r.k ?? defaultBM25Params.k
  r.b = r.b ?? defaultBM25Params.b
  r.d = r.d ?? defaultBM25Params.d
  return r as Required<BM25Params>
}
