<script>
  import SearchBox from '$lib/components/SearchBox.svelte';
  import FilterSidebar from '$lib/components/FilterSidebar.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import { searchStore } from '$lib/stores/search';
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';

  let showPreview = false;
  let selectedPDF = null;
  let scrolled = false;

  function handlePDFSelect(event) {
    selectedPDF = event.detail;
    showPreview = true;
  }

  function toggleFilters() {
    searchStore.update(current => ({ ...current, showFilters: !current.showFilters }));
  }

  onMount(() => {
    const handleScroll = () => {
      scrolled = window.scrollY > 90;
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  });

</script>

<main>
  <div class="large-logo" class:scrolled>
    <img src="/logo.png" alt="GovScape Logo" />
  </div>

  <h1 class="slogan">Search for 1+ Million .gov PDFs</h1>
  
  <SearchBox />
  <FilterSidebar />

  <button class="filter-button" on:click={toggleFilters} aria-label="Toggle Filters">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>
  </button>
  
  <div class="content-container">
    <div class="results-container">
      <ResultsGrid on:pdfSelect={handlePDFSelect} />
    </div>
  </div>

  {#if showPreview}
    <PDFPreview 
      show={showPreview}
      pdfData={selectedPDF}
      on:close={() => showPreview = false}
    />
  {/if}
</main>

<style>
  main {
    padding-top: 6rem;
    position: relative; /* For positioning the sidebar */
  }
  .filter-button {
    background: none;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 0.5rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .filter-button:hover {
    background-color: #f0f0f0;
  }
  
  .content-container {
    display: flex;
    padding: 0 5rem;
  }
  
  .results-container {
    flex: 1;
  }

  .slogan {
    text-align: center;
    padding: 1rem;
  }

  .large-logo {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 2rem;
    transition: opacity 0.3s ease, transform 0.3s ease;
    pointer-events: none;
  }

  .large-logo img {
    height: 120px;
    width: auto;
  }

  .large-logo.scrolled {
    opacity: 0;
    transform: translateY(-20px);
  }
</style>
