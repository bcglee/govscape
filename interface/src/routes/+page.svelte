<script>
  import SearchBox from '$lib/components/SearchBox.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import TypingEffect from '$lib/components/TypingEffect.svelte';
  import { onMount, onDestroy } from 'svelte';

  const govDomains = [
    'riversideca.gov',
    'tennille-ga.gov',
    'alabama.gov',
    'govinfo.gov',
    'sec.gov',
    'gpo.gov'
  ];

  let showPreview = false;
  let selectedPDF = null;
  let isSmallScreen = false;

  function handlePDFSelect(event) {
    selectedPDF = event.detail;
    showPreview = true;
  }

  function handleClosePreview() {
    showPreview = false;
    selectedPDF = null;
  }

  onMount(() => {
    function checkScreenSize() {
      isSmallScreen = window.innerWidth < 768;
    }
    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);

    return () => {
      window.removeEventListener('resize', checkScreenSize);
    };
  });
</script>

<main>
  <div class="title-container {isSmallScreen ? 'small-screen' : ''}">
    <h1>
      {#if isSmallScreen}
        Search 1+ Million PDFs across <TypingEffect words={govDomains} />
      {:else}
        Search 1+ Million PDFs<br />across <TypingEffect words={govDomains} />
      {/if}
    </h1>
  </div>
  <SearchBox />
  <ResultsGrid on:pdfSelect={handlePDFSelect} />
  <PDFPreview 
    show={showPreview}
    pdfData={selectedPDF}
    on:close={handleClosePreview}
  />
</main>

<style>
  main {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 150px;
  }

  .title-container {
    width: 550px;
    max-width: 100vw;
    padding: 2rem;
    margin-bottom: 1rem;
    white-space: nowrap;
  }

  .title-container.small-screen {
    white-space: normal;
  }

  .title-container h1 {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.35;
  }
</style>
