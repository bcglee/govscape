<script>
  import { searchStore } from '../stores/search';
  
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
    searchStore.update(state => ({
      ...state,
      filters: {
        dateFrom: dateFrom || null,
        dateTo: dateTo || null,
        minPages: minPages ? parseInt(minPages) : null,
        maxPages: maxPages ? parseInt(maxPages) : null,
        domain: selectedDomain || null
      }
    }));
  }
  
  function resetFilters() {
    dateFrom = '';
    dateTo = '';
    minPages = '';
    maxPages = '';
    selectedDomain = '';
    updateFilters();
  }
</script>

<div class="filter-sidebar">
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
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    width: 100%;
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
</style> 