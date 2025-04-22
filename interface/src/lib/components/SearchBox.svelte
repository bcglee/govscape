<script>
  import { searchStore } from '$lib/stores/search';
  import { search } from '$lib/api/search';

  let query = '';

  async function onSearch() {
    if (!query.trim()) return;

    searchStore.update(store => ({ ...store, loading: true, error: null }));

    try {
      const filters = $searchStore.filters;
      const { success, data, error } = await search(query, filters);

      if (!success) throw new Error(error);

      searchStore.update(store => ({ ...store, results: data.results, loading: false }));
    } catch (err) {
      console.error('Search error:', err);
      searchStore.update(store => ({ ...store, error: err.message, loading: false, results: [] }));
    }
  }
</script>

<div class="search-container">
  <form on:submit|preventDefault={onSearch}>
    <div class="search-input-wrapper">
      <input 
        type="text" 
        bind:value={query} 
        placeholder="Explore PDFs using natural language search..."
        aria-label="Search input"
      />
      <button type="submit" disabled={$searchStore.loading}>
        Search
      </button>
    </div>
  </form>
</div>

<style>
  .search-container {
    margin-bottom: 1rem;
    padding: 0 30%;
    min-width: 500px;
  }
  
  .search-input-wrapper {
    display: flex;
  }
  
  input {
    flex-grow: 1;
    padding: 0.5rem;
    border: 1px solid #ccc;
    border-radius: 4px 0 0 4px;
    outline: none;
  }
  
  button {
    padding: 0.5rem 1rem;
    background-color: #4a4a4a;
    color: white;
    border: none;
    border-radius: 0 4px 4px 0;
    cursor: pointer;
  }
  
  button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }
</style>