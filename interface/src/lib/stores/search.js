import { writable, get } from 'svelte/store';
import { apiFetch, IMAGE_BASE_URL } from '../utils/fetch';

export const searchStore = writable({
  query: '',
  results: [],
  filters: {},
  showFilters: false,
  loading: false,
  error: null
});

export const searchActions = {
  setQuery: (query) => {
    searchStore.update(store => ({
      ...store,
      query
    }));
  },
  
  toggleFilters: () => {
    searchStore.update(store => ({
      ...store,
      showFilters: !store.showFilters
    }));
  },
  
  updateFilters: (newFilters) => {
    searchStore.update(store => ({
      ...store,
      filters: {
        ...store.filters,
        ...newFilters
      }
    }));
  },
  
  clearResults: () => {
    searchStore.update(store => ({
      ...store,
      results: [],
      error: null
    }));
  },
  
  reset: () => {
    searchStore.set({
      query: '',
      filters: {},
      results: [],
      loading: false,
      error: null,
      showFilters: false
    });
  },
  
  performSearch: async () => {
    const currentStore = get(searchStore);
    const { query, filters } = currentStore;

    searchStore.update(store => ({ 
      ...store, 
      loading: true, 
      error: null,
      showFilters: false
    }));

    try {
      const responseData = await apiFetch('/search/', { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query, filters }) 
      }); 
      
      const results = (responseData.results || []).map(result => ({
        ...result,
        jpeg: result.jpeg && typeof result.jpeg === 'string' 
              ? `${IMAGE_BASE_URL}/${result.jpeg.split('/').slice(-2).join('/')}` 
              : null
      }));

      searchStore.update(store => ({
        ...store,
        results: results,
        loading: false
      }));
    } catch (err) {
      console.error('Search error in store:', err);
      searchStore.update(store => ({
        ...store,
        error: err.message || 'Search failed',
        loading: false,
        results: []
      }));
    }
  }
};
