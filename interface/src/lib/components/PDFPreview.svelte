<script>
  import { createEventDispatcher } from 'svelte';
  import { apiFetch, IMAGE_BASE_URL } from '../utils/fetch';

  export let show = false;
  export let pdfData = null;

  const dispatch = createEventDispatcher();

  let images = [];
  let loading = false;
  let error = null;
  let currentPage = 1;
  let totalPages = 0;

  async function fetchImages() {
    if (!pdfData?.id) return;

    loading = true;
    error = null;
    images = [];

    try {
      const data = await apiFetch(`/pages/${pdfData.id}`, { method: 'GET' });
      images = (data.images || []).map(img => {
        const parts = img.split('/');
        return `${IMAGE_BASE_URL}/${parts.slice(-2).join('/')}`;
      });
      totalPages = images.length;
      currentPage = 1;
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  $: if (show && pdfData) {
    fetchImages();
  }

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
</script>

{#if show}
  <div class="modal-backdrop" on:click={closeModal}>
    <div class="modal-content" on:click|stopPropagation>
      <div class="modal-header">
        <h5 class="modal-title">{pdfData?.id?.split('/').pop() || ''}</h5>
        <button class="btn-close" on:click={closeModal}></button>
      </div>
      <div class="modal-body">
        {#if loading}
          <p>Loading images...</p>
        {:else if error}
          <p style="color:red">{error}</p>
        {:else if images.length > 0}
          <div class="pdf-pages pdf-preview-grid">
            <div class="pdf-main-panel compact">
              <div class="pdf-image-wrapper compact">
                <img src={images[currentPage-1]} alt={`Page ${currentPage}`} class="pdf-main-image compact" />
              </div>
              <div class="page-navigation">
                <button class="carousel-nav prev" on:click={prevPage} disabled={currentPage === 1}>
                  <i class="bi bi-chevron-left"></i>
                </button>
                <span class="page-number">
                  Page {currentPage} of {totalPages}
                </span>
                <button class="carousel-nav next" on:click={nextPage} disabled={currentPage === totalPages}>
                  <i class="bi bi-chevron-right"></i>
                </button>
              </div>
            </div>
            <aside class="pdf-side-panel">
              <div class="metadata-box compact no-border">
                <div><b>ID:</b> {pdfData?.id || '-'}</div>
                <div><b>Page:</b> {currentPage} / {totalPages}</div>
                <div><b>Subdomain:</b> {pdfData?.subdomain || 'epa.gov'}</div>
                <div><b>Publish Date:</b> {pdfData?.publish_date || '2022-01-01'}</div>
                {#if pdfData?.title}
                  <div><b>Title:</b> {pdfData.title}</div>
                {/if}
                {#if pdfData?.agency}
                  <div><b>Agency:</b> {pdfData.agency}</div>
                {/if}
              </div>
              <div class="pdf-thumbnails-panel large">
                <h6>All Pages</h6>
                <div class="all-images all-pages-grid">
                  {#each images as img, i}
                    <img src={img} alt={`Page ${i+1}`} class:active-thumb={i+1===currentPage} on:click={() => currentPage=i+1} />
                  {/each}
                </div>
              </div>
            </aside>
          </div>
        {:else}
          <p>No images found.</p>
        {/if}
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
    width: 80%;
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
  .pdf-pages {
    width: 100%;
  }
  .pdf-preview-grid {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 0;
    align-items: flex-start;
    width: 100%;
    min-height: 60vh;
    justify-items: center;
  }
  .pdf-main-panel.compact {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    max-width: 480px;
    justify-content: center;
  }
  .pdf-side-panel {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    gap: 1.2rem;
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
  .pdf-image-wrapper {
    overflow: hidden;
    border-radius: 4px;
    border: 1px solid #ccc;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  .pdf-main-image {
    width: 100%;
    height: auto;
  }
  .metadata-box {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .metadata-box.compact {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 2px 16px 0 rgba(60, 80, 120, 0.10);
    padding: 1.1rem 1.3rem;
    font-size: 1.04rem;
    color: #1a2a3a;
    margin-bottom: 0.5rem;
    border: 1px solid #e6eaf2;
  }
  .metadata-box.compact.no-border {
    background: none;
    border: none;
    box-shadow: none;
    padding: 0 0 1.1rem 0;
    font-size: 1.04rem;
    color: #1a2a3a;
    margin-bottom: 0.5rem;
  }
  .metadata-box.compact.no-border h6 {
    margin: 0 0 0.7rem 0;
    font-size: 1.13rem;
    color: #1a2a3a;
    font-weight: 700;
    letter-spacing: 0.01em;
  }
  .pdf-thumbnails-panel.large {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 16px 0 rgba(60, 80, 120, 0.10);
    padding: 1.2rem 0.7rem;
    margin-top: 0.2rem;
    flex: 1 1 0;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    min-height: 0;
    border: 1px solid #e6eaf2;
  }
  .pdf-thumbnails-panel.large h6 {
    font-size: 1.08rem;
    color: #1a2a3a;
    font-weight: 700;
    margin-bottom: 0.8rem;
    letter-spacing: 0.01em;
  }
  .all-pages-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(88px, 1fr));
    gap: 18px;
    align-items: start;
    width: 100%;
    height: 100%;
    max-height: none;
    overflow-y: auto;
    padding-right: 4px;
    justify-items: center;
    background: #fff;
    border-radius: 8px;
    box-shadow: none;
  }
  .all-pages-grid img {
    width: 76px;
    height: 104px;
    object-fit: cover;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
    background: #fafdff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    cursor: pointer;
    transition: border 0.18s, box-shadow 0.18s, transform 0.18s, background 0.18s;
    filter: drop-shadow(0 1px 2px rgba(0,0,0,0.04));
  }
  .all-pages-grid img.active-thumb {
    border: 2.5px solid #007bff;
    box-shadow: 0 4px 16px rgba(0,123,255,0.13);
    transform: scale(1.10) translateY(-2px);
    background: #eaf6ff;
    z-index: 2;
  }
</style>
