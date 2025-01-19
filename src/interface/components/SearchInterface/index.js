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
            naturalMode: document.getElementById('naturalMode')
        };
        
        this.state = {
            currentOffset: 0,
            limit: 20,
            loading: false,
            sidebarActive: false,
            searchMode: 'keyword'
        };
        
        this.initializeEventListeners();
        this.initializeMasonry();
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

            const response = await API.searchCollections(
                query, 
                append ? this.state.currentOffset : 0,
                this.state.limit,
                this.state.searchMode
            );

            this.displayResults(response, append);
            this.state.currentOffset = response.pagination.offset + response.pagination.limit;
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
        }
        
        // Create and prepare all items before adding to DOM
        const fragment = document.createDocumentFragment();
        const items = response.results.map(result => this.createResultItem(result));
        
        items.forEach(item => fragment.appendChild(item));
        this.elements.resultsGrid.appendChild(fragment);

        imagesLoaded(this.elements.resultsGrid, () => {
            this.masonry.reloadItems();
            this.masonry.layout();
        });
    }

    createResultItem(result) {
        const item = document.createElement('div');
        item.className = 'grid-item';
        
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
                            <span class="file-type">${result.metadata.fileType}</span> • 
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
        if (this.state.loading) return;

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
} 