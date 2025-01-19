class SearchInterface {
    constructor() {
        this.elements = {
            searchInput: document.getElementById('searchInput'),
            searchBtn: document.getElementById('searchBtn'),
            luckyBtn: document.getElementById('luckyBtn'),
            imageInput: document.getElementById('imageInput'),
            resultsGrid: document.getElementById('resultsGrid')
        };
        
        this.initializeEventListeners();
        this.initializeMasonry();
    }

    initializeEventListeners() {
        const { searchBtn, luckyBtn, imageInput, searchInput } = this.elements;
        
        searchBtn.addEventListener('click', () => this.performSearch());
        luckyBtn.addEventListener('click', () => this.feelingLucky());
        imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performSearch();
        });
    }

    initializeMasonry() {
        this.masonry = new Masonry(this.elements.resultsGrid, {
            itemSelector: '.grid-item',
            columnWidth: '.grid-item',
            gutter: 20
        });
    }

    async performSearch() {
        const query = this.elements.searchInput.value.trim();
        if (!query) return;

        try {
            const results = await API.mockAPICall(query);
            this.displayResults(results);
        } catch (error) {
            console.error('Search failed:', error);
        }
    }

    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            const response = await API.uploadImage(file);
            console.log('Image upload response:', response);
        } catch (error) {
            console.error('Image upload failed:', error);
        }
    }

    feelingLucky() {
        // Implement "I'm Feeling Lucky" functionality
        console.log('Feeling Lucky clicked');
    }

    displayResults(results) {
        this.elements.resultsGrid.innerHTML = '';
        
        results.forEach(result => {
            const item = this.createResultItem(result);
            this.elements.resultsGrid.appendChild(item);
        });

        this.masonry.reloadItems();
        this.masonry.layout();
    }

    createResultItem(result) {
        const item = document.createElement('div');
        item.className = 'grid-item';
        item.innerHTML = `
            <img src="${result.imageUrl}" alt="${result.title}">
            <div class="p-3">
                <h5>${result.title}</h5>
            </div>
        `;
        return item;
    }
}

// Initialize the search interface when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SearchInterface();
}); 