<script>
  import SearchBox from '$lib/components/SearchBox.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import TypingEffect from '$lib/components/TypingEffect.svelte';

  const govDomains = [
    'riversideca.gov',
    'tennille-ga.gov',
    'sos.alabama.gov',
    'govinfo.gov',
    'sec.gov',
    'gpo.gov'
  ];

  let showPreview = false;
  let selectedPDF = null;

  function handlePDFSelect(event) {
    selectedPDF = event.detail;
    showPreview = true;
  }
</script>

<main>
  <div class="title-container">
    <h1>Search 1+ Million PDFs<br />across <TypingEffect words={govDomains} /></h1>
  </div>

  <SearchBox />

  <ResultsGrid on:pdfSelect={handlePDFSelect} />

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
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 150px;
  }

  .title-container {
    width: 550px;
    padding: 2rem;
    margin-bottom: 1rem;
    white-space: nowrap;
  }
  .title-container h1 {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.35;
  }
</style>
