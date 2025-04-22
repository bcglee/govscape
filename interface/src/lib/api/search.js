import { apiFetch } from '../utils/fetch';

export async function search(query, filters = {}) {
    try {
        const data = await apiFetch('/search/', {
            body: JSON.stringify({ query, filters }),
        });

        return {
            success: true,
            data,
        };
    } catch (error) {
        console.error('Search error:', error);
        return {
            success: false,
            error: error.message,
        };
    }
}
