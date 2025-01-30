class SearchInterface {
    constructor() {
        this.elements = {
            searchInput: document.getElementById('searchInput'),
            searchBtn: document.getElementById('searchBtn'),
            resultsGrid: document.getElementById('resultsGrid'),
            toggleFilter: document.getElementById('toggleFilter'),
            filterSidebar: document.getElementById('filterSidebar'),
            closeFilter: document.getElementById('closeFilter'),
            sidebarOverlay: document.getElementById('sidebarOverlay'),
            keywordMode: document.getElementById('keywordMode'),
            naturalMode: document.getElementById('naturalMode'),
            pdfPreviewModal: document.getElementById('pdfPreviewModal'),
            pageCarousel: document.querySelector('.page-carousel'),
            prevButton: document.querySelector('.carousel-nav.prev'),
            nextButton: document.querySelector('.carousel-nav.next'),
            dateFrom: document.getElementById('dateFrom'),
            dateTo: document.getElementById('dateTo'),
            pagesRangeMin: document.getElementById('pagesRangeMin'),
            pagesRangeMax: document.getElementById('pagesRangeMax'),
            pagesMin: document.getElementById('pagesMin'),
            pagesMax: document.getElementById('pagesMax'),
            applyFilters: document.getElementById('applyFilters'),
            resetFilters: document.getElementById('resetFilters')
        };
        
        this.state = {
            currentOffset: 0,
            limit: 20,
            loading: false,
            sidebarActive: false,
            searchMode: 'keyword',
            filters: {
                dateRange: { from: null, to: null },
                subdomains: ['gov'],
                pages: { min: 0, max: 500 },
                sizes: []
            }
        };
        
        this.modal = new bootstrap.Modal(this.elements.pdfPreviewModal);
        this.currentPdfData = null;
        
        this.initializeEventListeners();
        this.initializeMasonry();
        this.initializeFilters();
        this.loadInitialResults();
    }

    initializeEventListeners() {
        const { 
            searchBtn, 
            searchInput, 
            toggleFilter, 
            closeFilter, 
            sidebarOverlay,
            keywordMode,
            naturalMode 
        } = this.elements;
        
        searchBtn.addEventListener('click', () => this.performSearch());
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performSearch();
        });
        
        keywordMode.addEventListener('change', () => this.updateSearchMode('keyword'));
        naturalMode.addEventListener('change', () => this.updateSearchMode('natural'));

        toggleFilter.addEventListener('click', () => this.toggleSidebar());
        closeFilter.addEventListener('click', () => this.toggleSidebar());
        sidebarOverlay.addEventListener('click', () => this.toggleSidebar());

        window.addEventListener('scroll', () => this.handleScroll());
    }

    updateSearchMode(mode) {
        this.state.searchMode = mode;
        const input = this.elements.searchInput;
        
        if (mode === 'keyword') {
            input.placeholder = 'Enter keywords to search...';
        } else {
            input.placeholder = 'Describe what you want to find...';
        }
    }

    async performSearch(append = false) {
        if (this.state.loading) return;
        
        const query = this.elements.searchInput.value.trim();
        if (!query) return;

        try {
            this.state.loading = true;
            this.updateLoadingState(true);

            // Fade out current results
            if (!append) {
                this.elements.resultsGrid.classList.add('fade-out');
                await new Promise(resolve => setTimeout(resolve, 400));
            }

            const response = await API.searchCollections(
                query, 
                append ? this.state.currentOffset : 0,
                this.state.limit,
                this.state.searchMode
            );

            this.displayResults(response, append);
            this.state.currentOffset = response.pagination.offset + response.pagination.limit;

            // Fade in new results
            if (!append) {
                this.elements.resultsGrid.classList.remove('fade-out');
                this.elements.resultsGrid.classList.add('fade-in');
            }

        } catch (error) {
            console.error('Search failed:', error);
        } finally {
            this.state.loading = false;
            this.updateLoadingState(false);
        }
    }

    displayResults(response, append = false) {
        if (!append) {
            this.elements.resultsGrid.innerHTML = '';
            this.elements.resultsGrid.classList.remove('fade-in');
        }
        
        // Create and prepare all items before adding to DOM
        const fragment = document.createDocumentFragment();
        const items = response.results.map((result, index) => {
            const item = this.createResultItem(result);
            item.style.animationDelay = `${index * 50}ms`;
            return item;
        });
        
        items.forEach(item => fragment.appendChild(item));
        this.elements.resultsGrid.appendChild(fragment);

        imagesLoaded(this.elements.resultsGrid, () => {
            this.masonry.reloadItems();
            this.masonry.layout();
            requestAnimationFrame(() => {
                this.elements.resultsGrid.classList.add('fade-in');
            });
        });
    }

    createResultItem(result) {
        const item = document.createElement('div');
        item.className = 'grid-item';
        
        item.addEventListener('click', (e) => {
            e.preventDefault();
            this.showPdfPreview(result);
        });
        
        item.innerHTML = `
            <a href="${result.url}" target="_blank" class="result-link">
                <img src="${result.image.thumbnail_url}" 
                     alt="${result.title}"
                     loading="lazy">
                <div class="info">
                    <h3 class="title">${result.title}</h3>
                    <div class="metadata">
                        <div class="agency">${result.metadata.creator}</div>
                        <div class="file-info">
                            <span class="year">${result.metadata.date}</span> • 
                            <span class="file-size">${result.metadata.fileSize}</span> • 
                            <span class="pages">${result.metadata.pages} pages</span>
                        </div>
                    </div>
                </div>
            </a>
        `;

        return item;
    }

    handleScroll() {
        // Only load more if we're not loading and have a search query
        if (this.state.loading || !this.elements.searchInput.value.trim()) return;

        const { scrollTop, scrollHeight, clientHeight } = document.documentElement;
        if (scrollTop + clientHeight >= scrollHeight - 300) {
            this.performSearch(true);
        }
    }

    updateLoadingState(loading) {
        const loadingEl = document.querySelector('.loading-indicator') || 
                         this.createLoadingIndicator();
        loadingEl.style.display = loading ? 'block' : 'none';
    }

    createLoadingIndicator() {
        const loadingEl = document.createElement('div');
        loadingEl.className = 'loading-indicator text-center mt-4';
        loadingEl.innerHTML = '<div class="spinner-border text-primary" role="status"></div>';
        this.elements.resultsGrid.parentNode.appendChild(loadingEl);
        return loadingEl;
    }

    toggleSidebar() {
        const { filterSidebar, sidebarOverlay } = this.elements;
        this.state.sidebarActive = !this.state.sidebarActive;
        
        filterSidebar.classList.toggle('active');
        sidebarOverlay.classList.toggle('active');
        document.querySelector('main').classList.toggle('sidebar-active');
        
        // Prevent body scroll when sidebar is open on mobile
        if (window.innerWidth <= 768) {
            document.body.style.overflow = this.state.sidebarActive ? 'hidden' : '';
        }
    }

    async loadInitialResults() {
        try {
            this.state.loading = true;
            this.updateLoadingState(true);
            
            const response = await API.searchCollections('', 0, 20);
            this.displayResults(response);
            // Reset offset after loading initial results
            this.state.currentOffset = 0;
        } catch (error) {
            console.error('Failed to load initial results:', error);
        } finally {
            this.state.loading = false;
            this.updateLoadingState(false);
        }
    }

    initializeMasonry() {
        this.masonry = new Masonry(this.elements.resultsGrid, {
            itemSelector: '.grid-item',
            columnWidth: 300,
            gutter: 20,
            fitWidth: true,
            transitionDuration: 0,  // Disable animation during layout
            horizontalOrder: true,
            initLayout: false
        });
    }

    showPdfPreview(result) {
        this.currentPdfData = result;
        const modal = this.elements.pdfPreviewModal;
        
        // Update modal title
        modal.querySelector('.modal-title').textContent = result.title;
        
        // Update metadata
        const metadata = modal.querySelector('.pdf-metadata');
        metadata.innerHTML = `
            <dl>
                <dt>Department</dt>
                <dd>${result.metadata.creator}</dd>
                <dt>Date</dt>
                <dd>${result.metadata.date}</dd>
                <dt>File Size</dt>
                <dd>${result.metadata.fileSize}</dd>
                <dt>Pages</dt>
                <dd>${result.metadata.pages}</dd>
                <dt>Keywords</dt>
                <dd>${result.metadata.keywords.join(', ')}</dd>
            </dl>
        `;
        
        // Update text content
        modal.querySelector('.pdf-text').innerHTML = `
            <h6>Abstract</h6>
            <p>${result.metadata.abstract}</p>
        `;
        
        // Initialize carousel
        this.initializePageCarousel(result);
        
        // Update download link
        modal.querySelector('.download-pdf').href = result.url;
        
        this.modal.show();
    }

    initializePageCarousel(result) {
        const carousel = this.elements.pageCarousel;
        const pages = result.metadata.pageImages;
        const matchedPageIndex = Math.floor(pages.length / 2);
        
        // Update page numbers
        const modal = this.elements.pdfPreviewModal;
        modal.querySelector('.current-page').textContent = matchedPageIndex + 1;
        modal.querySelector('.total-pages').textContent = pages.length;
        
        // Calculate visible pages
        const totalVisible = 5;
        const start = Math.max(0, matchedPageIndex - Math.floor(totalVisible / 2));
        
        carousel.innerHTML = `
            <div class="page-carousel-inner">
                ${pages.map((page, i) => `
                    <div class="page-item ${i === matchedPageIndex ? 'active' : ''}"
                        data-page="${i + 1}"
                        style="transform: translateX(-${start * 20}%)">
                        <img src="${page.thumbnailUrl}" alt="Page ${page.pageNumber}">
                    </div>
                `).join('')}
            </div>
        `;
        
        // Add click handlers for thumbnails
        carousel.querySelectorAll('.page-item').forEach((item, i) => {
            if (i !== matchedPageIndex) {
                item.addEventListener('click', () => {
                    this.goToPage(i);
                });
            }
        });
        
        this.initializeCarouselControls();
    }

    initializeCarouselControls() {
        const { prevButton, nextButton, pageCarousel } = this.elements;
        const inner = pageCarousel.querySelector('.page-carousel-inner');
        let currentIndex = Math.floor(this.currentPdfData.metadata.pageImages.length / 2);
        
        const goToPage = (index) => {
            const items = pageCarousel.querySelectorAll('.page-item');
            const totalVisible = 5;
            
            items.forEach((item, i) => {
                item.classList.remove('active');
                if (i === index) {
                    item.classList.add('active');
                }
            });
            
            // Calculate new transform
            const start = Math.max(0, index - Math.floor(totalVisible / 2));
            inner.style.transform = `translateX(-${start * 20}%)`;
            
            // Update page number
            currentIndex = index;
            this.elements.pdfPreviewModal.querySelector('.current-page').textContent = index + 1;
        };
        
        this.goToPage = goToPage;  // Make it available for thumbnail clicks
        
        prevButton.onclick = () => {
            if (currentIndex > 0) {
                goToPage(currentIndex - 1);
            }
        };
        
        nextButton.onclick = () => {
            if (currentIndex < this.currentPdfData.metadata.pageImages.length - 1) {
                goToPage(currentIndex + 1);
            }
        };
    }

    initializeFilters() {
        // Initialize range slider
        this.elements.pagesRangeMin.addEventListener('input', (e) => {
            const value = e.target.value;
            this.elements.pagesMin.textContent = value;
            this.elements.pagesRangeMax.min = value;
            this.updateRangeSelection();
        });
        
        this.elements.pagesRangeMax.addEventListener('input', (e) => {
            const value = e.target.value;
            this.elements.pagesMax.textContent = value;
            this.elements.pagesRangeMin.max = value;
            this.updateRangeSelection();
        });

        // Initial range selection update
        this.updateRangeSelection();
        
        // Apply filters
        this.elements.applyFilters.addEventListener('click', () => {
            this.updateFilters();
            this.performSearch();
            this.toggleSidebar();  // Always close the panel after applying filters
        });
        
        // Reset filters
        this.elements.resetFilters.addEventListener('click', () => {
            this.resetFilters();
            this.toggleSidebar();  // Also close the panel after resetting filters
        });
    }

    updateRangeSelection() {
        const rangeSelected = document.querySelector('.range-selected');
        const min = parseInt(this.elements.pagesRangeMin.value);
        const max = parseInt(this.elements.pagesRangeMax.value);
        const range = 500; // total range
        
        const left = (min / range) * 100;
        const right = 100 - (max / range) * 100;
        
        rangeSelected.style.left = left + '%';
        rangeSelected.style.right = right + '%';
    }

    updateFilters() {
        this.state.filters = {
            dateRange: {
                from: this.elements.dateFrom.value,
                to: this.elements.dateTo.value
            },
            subdomains: Array.from(document.querySelectorAll('.subdomain-options input:checked'))
                .map(input => input.value),
            pages: {
                min: parseInt(this.elements.pagesRangeMin.value),
                max: parseInt(this.elements.pagesRangeMax.value)
            },
            sizes: Array.from(document.querySelectorAll('.size-options input:checked'))
                .map(input => input.value)
        };
    }

    resetFilters() {
        // Reset all form elements
        this.elements.dateFrom.value = '';
        this.elements.dateTo.value = '';
        this.elements.pagesRangeMin.value = 0;
        this.elements.pagesRangeMax.value = 500;
        this.elements.pagesMin.textContent = '0';
        this.elements.pagesMax.textContent = '500';
        
        document.querySelectorAll('.subdomain-options input').forEach(input => {
            input.checked = input.value === 'gov';
        });
        
        document.querySelectorAll('.size-options input').forEach(input => {
            input.checked = false;
        });
        
        // Reset state
        this.state.filters = {
            dateRange: { from: null, to: null },
            subdomains: ['gov'],
            pages: { min: 0, max: 500 },
            sizes: []
        };
        
        // Perform search with reset filters
        this.performSearch();
    }
} 