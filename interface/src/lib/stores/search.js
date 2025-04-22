import { writable } from 'svelte/store';

export const searchStore = writable({
  query: '',
  results: [],
  filters: {},
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
  
  updateFilters: (filters) => {
    searchStore.update(store => ({
      ...store,
      filters: {
        ...store.filters,
        ...filters
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
  }
};