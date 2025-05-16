<script>
  import { searchStore, searchActions } from '$lib/stores/search';

  let query = '';
  let searchMode = 'natural'; // 'natural' or 'keywords'

  searchStore.subscribe(store => {
    if (query !== store.query) {
      query = store.query;
    }
  });

  function setMode(mode) {
    searchMode = mode;
    query = '';
    searchActions.setQuery('');
  }

  async function onSearch() {
    if (!query.trim()) {
      searchActions.clearResults();
      return;
    }

    searchActions.setQuery(query);
    searchActions.performSearch();
  }
</script>

<div class="search-container">
  <div class="search-tabs">
    <button type="button" class:active-tab={searchMode==='natural'} on:click={() => setMode('natural')}>Natural Language</button>
    <button type="button" class:active-tab={searchMode==='keywords'} on:click={() => setMode('keywords')}>Keywords</button>
  </div>
  <form on:submit|preventDefault={onSearch}>
    <div class="search-input-wrapper">
      <input 
        type="text" 
        bind:value={query} 
        placeholder={searchMode==='natural' ? 'Explore PDFs using natural language search...' : 'Enter keywords, separated by commas'}
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
    margin-bottom: 3rem;
    padding: 0 30%;
    min-width: 500px;
  }
  .search-tabs {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 0;
    gap: 0;
    position: relative;
    z-index: 2;
  }
  .search-tabs button {
    background: #f4f4f4;
    border: 1px solid #ccc;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 0.5rem 1.2rem;
    font-size: 1rem;
    cursor: pointer;
    color: #333;
    outline: none;
    transition: background 0.18s, color 0.18s;
    margin-right: 2px;
  }
  .search-tabs button:last-child {
    margin-right: 0;
  }
  .search-tabs button.active-tab {
    background: #fff;
    color: #007bff;
    border-bottom: 1.5px solid #fff;
    font-weight: bold;
    z-index: 2;
  }
  .search-input-wrapper {
    display: flex;
    position: relative;
    top: -1px;
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
