<script>
  import { createEventDispatcher, onMount, afterUpdate } from 'svelte';
  import Masonry from 'masonry-layout';
  import imagesLoaded from 'imagesloaded';
  import { searchStore } from '$lib/stores/search';

  const dispatch = createEventDispatcher();
  let gridElement;
  let masonry;

  // Subscribe to searchStore.results
  $: results = $searchStore.results;

  function handlePDFSelect(pdf, page) {
    const pdfId = pdf.split('/').pop();

    dispatch('pdfSelect', { pdf, page, id: pdfId });
  }

  function getAgencyName(subdomain) {
    if (!subdomain) return '';
    return subdomain;
  }

  function handleImageError(event, result) {
    console.error('Image failed to load:', {
      src: event.target.src,
      result: result
    });
  }

  onMount(() => {
    if (typeof window !== 'undefined' && gridElement) {
      masonry = new Masonry(gridElement, {
        itemSelector: '.grid-item',
        columnWidth: '.grid-sizer',
        percentPosition: true,
        gutter: 20
      });

      imagesLoaded(gridElement).on('progress', () => {
        masonry?.layout();
      });
    }
  });

  afterUpdate(() => {
    if (masonry && results.length > 0) {
      setTimeout(() => {
        masonry.reloadItems();
        masonry.layout();
      }, 100);
    }
  });
</script>

<div class="results-grid" bind:this={gridElement}>
  {#if results.length > 0}
    <div class="grid-sizer"></div>
    {#each results as result}
      <div class="grid-item">
        <div class="result-card" on:click={() => handlePDFSelect(result.pdf, result.page)}>
          <div class="image-container">
            <img 
              src={result.jpeg} 
              alt={`PDF Page ${result.page}`}
              loading="lazy"
              on:error={(e) => handleImageError(e, result)}
            />
          </div>
          <div class="result-info">
            <h5>{result.pdf.split('/').pop()}</h5>
            <p>Page: {result.page}</p>
            {#if result.subdomain}
              <div class="agency-info">
                <p class="subdomain">{result.subdomain}</p>
                <p class="agency-name">{getAgencyName(result.subdomain)}</p>
              </div>
            {/if}
          </div>
        </div>
      </div>
    {/each}
  {/if}
</div>

<style>
  .results-grid {
    width: 100%;
    margin: 0 auto;
  }

  .grid-sizer,
  .grid-item {
    width: calc(33.333% - 20px);
    margin-bottom: 20px;
  }

  @media (max-width: 768px) {
    .grid-sizer,
    .grid-item {
      width: calc(50% - 20px);
    }
  }

  @media (max-width: 480px) {
    .grid-sizer,
    .grid-item {
      width: 100%;
    }
  }

  .result-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    transition: transform 0.2s;
    overflow: hidden;
  }

  .result-card:hover {
    transform: translateY(-2px);
  }

  .image-container {
    width: 100%;
    height: 200px;
    overflow: hidden;
    margin-bottom: 15px;
  }

  .image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .result-info h5 {
    margin: 0 0 10px 0;
    font-size: 1.2rem;
    color: var(--text-color);
    word-wrap: break-word;
    word-break: break-word;
  }

  .result-info p {
    margin: 5px 0;
    color: #666;
    word-wrap: break-word;
    word-break: break-word;
  }

  .agency-info {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #eee;
  }

  .subdomain {
    color: #007bff;
    font-weight: bold;
  }

  .agency-name {
    color: #666;
    font-size: 0.9em;
  }
</style>
