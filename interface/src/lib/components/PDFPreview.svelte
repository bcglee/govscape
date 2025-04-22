<script>
  import { onMount } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  
  export let show = false;
  export let pdfData = null;
  
  const dispatch = createEventDispatcher();
  
  let currentPage = 1;
  let totalPages = 0;
  
  function closeModal() {
    show = false;
    dispatch('close');
  }
  
  function nextPage() {
    if (currentPage < totalPages) {
      currentPage++;
    }
  }
  
  function prevPage() {
    if (currentPage > 1) {
      currentPage--;
    }
  }
  
  $: if (pdfData) {
    totalPages = pdfData.totalPages || 0;
  }
</script>

{#if show}
  <div class="modal-backdrop" on:click={closeModal}>
    <div class="modal-content" on:click|stopPropagation>
      <div class="modal-header">
        <h5 class="modal-title">{pdfData?.title || ''}</h5>
        <button class="btn-close" on:click={closeModal}></button>
      </div>
      
      <div class="modal-body">
        <div class="pdf-preview-container">
          <div class="pdf-details">
            <div class="pdf-metadata">
              <p><i class="bi bi-building"></i> {pdfData?.agency || ''}</p>
              <p><i class="bi bi-globe"></i> {pdfData?.subdomain || ''}</p>
              <p><i class="bi bi-calendar"></i> {pdfData?.date || ''}</p>
            </div>
            <div class="pdf-text">
              {pdfData?.snippet || ''}
            </div>
          </div>
          
          <div class="pdf-pages">
            <div class="page-navigation">
              <button class="carousel-nav prev" on:click={prevPage}>
                <i class="bi bi-chevron-left"></i>
              </button>
              <span class="page-number">
                Page {currentPage} of {totalPages}
              </span>
              <button class="carousel-nav next" on:click={nextPage}>
                <i class="bi bi-chevron-right"></i>
              </button>
            </div>
            <div class="page-carousel">
              {#if pdfData?.pages}
                {#each pdfData.pages as page, i}
                  {#if i + 1 === currentPage}
                    <img src={page.url} alt={`Page ${i + 1}`} />
                  {/if}
                {/each}
              {/if}
            </div>
          </div>
        </div>
      </div>
      
      <div class="modal-footer">
        <a href={pdfData?.url} class="btn btn-primary" target="_blank">
          <i class="bi bi-download"></i> Download PDF
        </a>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1050;
  }
  
  .modal-content {
    background: white;
    width: 90%;
    max-width: 1200px;
    max-height: 90vh;
    border-radius: 4px;
    display: flex;
    flex-direction: column;
  }
  
  .modal-header {
    padding: 1rem;
    border-bottom: 1px solid #dee2e6;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .modal-body {
    padding: 1rem;
    overflow-y: auto;
  }
  
  .modal-footer {
    padding: 1rem;
    border-top: 1px solid #dee2e6;
  }
  
  .pdf-preview-container {
    display: grid;
    grid-template-columns: 300px 1fr;
    gap: 1rem;
  }
  
  .pdf-details {
    border-right: 1px solid #dee2e6;
    padding-right: 1rem;
  }
  
  .pdf-metadata {
    margin-bottom: 1rem;
  }
  
  .pdf-metadata p {
    margin-bottom: 0.5rem;
    color: #6c757d;
  }
  
  .page-navigation {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  
  .carousel-nav {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
  }
  
  .page-carousel {
    display: flex;
    justify-content: center;
  }
  
  .page-carousel img {
    max-width: 100%;
    height: auto;
  }
</style> 