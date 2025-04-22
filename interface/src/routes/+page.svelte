<script>
  import SearchBox from '$lib/components/SearchBox.svelte';
  import FilterSidebar from '$lib/components/FilterSidebar.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import { searchStore } from '$lib/stores/search';

  let showPreview = false;
  let selectedPDF = null;

  function handlePDFSelect(event) {
    selectedPDF = event.detail;
    showPreview = true;
  }
</script>

<main>
  <h1 class="slogan">Search for 1+ Million .gov PDFs</h1>
  
  <SearchBox />
  
  <div class="content-container">
    <div class="results-container">
      <ResultsGrid on:pdfSelect={handlePDFSelect} />
    </div>
    
    {#if $searchStore.showFilters}
      <FilterSidebar />
    {/if}
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
  }
  
  ul {
    list-style: none;
    padding: 0;
  }
  
  li {
    border-bottom: 1px solid #eee;
    padding: 1rem 0;
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
</style>
