<script>
  import { fade, slide } from 'svelte/transition';
  import { searchStore, searchActions } from '$lib/stores/search';
  export let show = false;

  let crawlDate = '2024';
  let subdomain = '';
  let pageCount = '';

  const crawlDateOptions = [
    { value: '2024', label: '2024' },
    { value: '2020', label: '2020' },
    { value: '2016', label: '2016' },
    { value: '2012', label: '2012' },
    { value: '2008', label: '2008' },
  ];

  const subdomainOptions = [
    { value: '', label: 'Any subdomain' },
    { value: 'epa.gov', label: 'epa.gov' },
    { value: 'nasa.gov', label: 'nasa.gov' },
    { value: 'cdc.gov', label: 'cdc.gov' },
  ];

  function applyFilters() {
    let minPages = null;
    let maxPages = null;

    if (pageCount) {
      const parts = pageCount.split('-');
      if (parts.length === 1 && pageCount.endsWith('+')) {
        minPages = parseInt(parts[0].slice(0, -1));
      } else if (parts.length === 1) {
        minPages = parseInt(parts[0]);
        maxPages = minPages; // Exact match if only one number
      } else if (parts.length === 2) {
        minPages = parts[0] ? parseInt(parts[0]) : null;
        maxPages = parts[1] ? parseInt(parts[1]) : null;
      }
    }

    searchActions.updateFilters({
      crawlDate: crawlDate || null,
      subdomain: subdomain || null,
      minPages: minPages,
      maxPages: maxPages,
    });
  }
</script>

{#if show}
  <div transition:slide={{ duration: 300 }}>
    <div class="advanced-search-container">
      <div class="advanced-search-title">Search Filters</div>
      <div class="filters-grid">
        <div class="filter-item">
          <label for="crawlDate">Crawl Date</label>
          <select id="crawlDate" bind:value={crawlDate} on:change={applyFilters}>
            {#each crawlDateOptions as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
        </div>
        <div class="filter-item">
          <label for="subdomain">Subdomain</label>
          <select id="subdomain" bind:value={subdomain} on:change={applyFilters}>
            {#each subdomainOptions as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
        </div>
        <div class="filter-item">
          <label for="pageCount">Page Count</label>
          <input type="text" id="pageCount" placeholder="Enter range" bind:value={pageCount} on:change={applyFilters} />
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .advanced-search-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 24px;
    padding: 16px;
    background: #fff;
    border-radius: 24px;
    box-shadow: 0 1px 6px rgba(32, 33, 36, 0.08);
  }

  .advanced-search-title {
    color: var(--text-color-primary);
    margin-bottom: 1rem;
    align-self: flex-start;
  }

  .filters-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    width: 100%;
  }

  .filter-item {
    display: flex;
    flex-direction: column;
  }

  .filter-item label {
    font-size: 0.8rem;
    color: var(--text-color-secondary);
    margin-bottom: 0.3rem;
  }

  .filter-item select,
  .filter-item input {
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #fff;
    font-size: 0.8rem;
    color: var(--text-color-primary);
    padding: 8px;
  }
  .filter-item input::placeholder {
    color: var(--text-color-secondary);
  }
</style>
