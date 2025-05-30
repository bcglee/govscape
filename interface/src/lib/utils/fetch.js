const IS_DEV = import.meta.env.DEV;

const ENDPOINTS = {
    DEV: {
        textual: 'http://localhost:8080',
        visual: 'http://localhost:8080',
        keyword: 'http://localhost:8080',
    },
    PROD: {
        textual: 'https://govscape.net/uae/api',
        visual: 'https://govscape.net/clip/api',
        keyword: 'https://govscape.net/uae/api', // TODO: update to api endpoint
    }
};

export const getApiBaseUrl = (searchMode = 'textual') => {
  if (IS_DEV) return ENDPOINTS.DEV[searchMode] + '/api';

  return ENDPOINTS.PROD[searchMode] + '/api';
};

export const getImageBaseUrl = (searchMode = 'textual') => {
  if (IS_DEV) return ENDPOINTS.DEV[searchMode] + '/images';
  
  return ENDPOINTS.PROD[searchMode] + '/images';
};

export async function apiFetch(endpoint, options = {}, searchMode = 'textual') {
    const defaultOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'include',
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };

    try {
        const apiUrl = getApiBaseUrl(searchMode);
        const response = await fetch(`${apiUrl}${endpoint}`, mergedOptions);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}
