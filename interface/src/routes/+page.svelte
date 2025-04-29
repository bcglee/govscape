<script>
  import SearchBox from '$lib/components/SearchBox.svelte';
  import FilterSidebar from '$lib/components/FilterSidebar.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import { searchStore } from '$lib/stores/search';
  import { onMount } from 'svelte';

  let showPreview = false;
  let selectedPDF = null;
  let scrolled = false;

  function handlePDFSelect(event) {
    selectedPDF = event.detail;
    showPreview = true;
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

  .large-logo {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 2rem;
    transition: opacity 0.3s ease, transform 0.3s ease;
  }

  .large-logo img {
    height: 120px;
    width: auto;
  }

  .large-logo.scrolled {
    opacity: 0;
    transform: translateY(-20px);
    pointer-events: none;
  }
</style>
