<script>
  import { searchStore, searchActions } from '../stores/search';
  
  let dateFrom = '';
  let dateTo = '';
  let minPages = '';
  let maxPages = '';
  let selectedDomain = '';
  
  const domains = [
    { value: '', label: 'All Domains' },
    { value: 'seattle.gov', label: 'Seattle.gov' },
    { value: 'kingcounty.gov', label: 'King County' },
    { value: 'wa.gov', label: 'Washington State' },
    { value: 'gov', label: 'Federal Government' }
  ];
  
  function updateFilters() {
    searchActions.updateFilters({
      dateFrom: dateFrom || null,
      dateTo: dateTo || null,
      minPages: minPages ? parseInt(minPages) : null,
      maxPages: maxPages ? parseInt(maxPages) : null,
      domain: selectedDomain || null
    });
  }
  
  function resetFilters() {
    dateFrom = '';
    dateTo = '';
    minPages = '';
    maxPages = '';
    selectedDomain = '';
    updateFilters();
  }

  function closeSidebar() {
    searchStore.update(current => ({ ...current, showFilters: false }));
  }
</script>

<div class="filter-sidebar" class:visible={$searchStore.showFilters}>
  <button class="close-sidebar-btn" on:click={closeSidebar} aria-label="Close filters">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
  </button>
  <div class="filter-section">
    <h3>Date Range</h3>
    <div class="date-inputs">
      <div class="input-group">
        <label for="dateFrom">From</label>
        <input
          type="date"
          id="dateFrom"
          bind:value={dateFrom}
          on:change={updateFilters}
        />
      </div>
      <div class="input-group">
        <label for="dateTo">To</label>
        <input
          type="date"
          id="dateTo"
          bind:value={dateTo}
          on:change={updateFilters}
        />
      </div>
    </div>
  </div>
  
  <div class="filter-section">
    <h3>Page Count</h3>
    <div class="page-inputs">
      <div class="input-group">
        <label for="minPages">Min</label>
        <input
          type="number"
          id="minPages"
          bind:value={minPages}
          min="1"
          on:change={updateFilters}
        />
      </div>
      <div class="input-group">
        <label for="maxPages">Max</label>
        <input
          type="number"
          id="maxPages"
          bind:value={maxPages}
          min="1"
          on:change={updateFilters}
        />
      </div>
    </div>
  </div>
  
  <div class="filter-section">
    <h3>Domain</h3>
    <select
      bind:value={selectedDomain}
      on:change={updateFilters}
      class="domain-select"
    >
      {#each domains as domain}
        <option value={domain.value}>{domain.label}</option>
      {/each}
    </select>
  </div>
  
  <button class="reset-btn" on:click={resetFilters}>
    Reset Filters
  </button>
</div>

<style>
  .filter-sidebar {
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;
    background: white;
    border-right: 1px solid #e0e0e0;
    box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    width: 350px;
    z-index: 1000;
    transform: translateX(-100%);
    transition: transform 0.3s ease-in-out;
    overflow-y: auto;
  }
  
  .filter-sidebar.visible {
    transform: translateX(0);
  }
  
  .filter-section {
    margin-bottom: 1.5rem;
  }
  
  .filter-section:last-child {
    margin-bottom: 1rem;
  }
  
  h3 {
    font-size: 1rem;
    color: #343a40;
    margin-bottom: 0.75rem;
    font-weight: 600;
  }
  
  .date-inputs,
  .page-inputs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }
  
  .input-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  
  label {
    font-size: 0.875rem;
    color: #6c757d;
  }
  
  input,
  select {
    padding: 0.5rem;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    font-size: 0.875rem;
    width: 100%;
  }
  
  input:focus,
  select:focus {
    outline: none;
    border-color: #007bff;
  }
  
  .domain-select {
    width: 100%;
  }
  
  .reset-btn {
    width: 100%;
    padding: 0.75rem;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    color: #495057;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .reset-btn:hover {
    background: #e9ecef;
    border-color: #ced4da;
  }

  .close-sidebar-btn {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    background: none;
    border: none;
    font-size: 1.5rem; /* Adjust size as needed */
    cursor: pointer;
    padding: 0.25rem;
    line-height: 1;
    color: #6c757d; /* Match other icon/text colors or choose new */
  }

  .close-sidebar-btn:hover {
    color: #343a40;
  }
</style>
