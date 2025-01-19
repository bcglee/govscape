class API {
    static async searchCollections(query, offset = 0, limit = 20, mode = 'keyword') {
        // Mock API response
        return new Promise(resolve => {
            setTimeout(() => {
                const results = Array.from({ length: 15 }, (_, i) => ({
                    id: `doc-${i + 1}`,
                    url: `https://example.gov/documents/${i + 1}.pdf`,
                    title: [
                        "Federal Guidelines for Cybersecurity Implementation",
                        "Annual Report on Environmental Protection Measures",
                        "National Infrastructure Development Plan",
                        "Public Health Policy Framework 2023",
                        "Department of Education Strategic Plan",
                        "Federal Emergency Response Protocol",
                        "Economic Impact Analysis Report",
                        "Climate Change Adaptation Strategy",
                        "Digital Government Transformation Guide",
                        "National Security Policy Review",
                        "Healthcare System Reform Proposal",
                        "Urban Development Guidelines",
                        "Renewable Energy Initiative Report",
                        "Federal Data Privacy Standards",
                        "Transportation Infrastructure Assessment"
                    ][i],
                    similarity_score: 0.9 - (i * 0.02),
                    metadata: {
                        text: "Official government document outlining policies, guidelines, and strategic plans",
                        date: `2${(20 + Math.floor(i/3)).toString().padStart(2, '0')}`,
                        creator: [
                            "Department of Homeland Security",
                            "Environmental Protection Agency",
                            "Department of Transportation",
                            "Department of Health and Human Services",
                            "Department of Education",
                            "Federal Emergency Management Agency",
                            "Department of Treasury",
                            "Environmental Protection Agency",
                            "Office of Management and Budget",
                            "Department of Defense",
                            "Centers for Medicare & Medicaid Services",
                            "Department of Housing and Urban Development",
                            "Department of Energy",
                            "Federal Trade Commission",
                            "Department of Transportation"
                        ][i],
                        fileType: "PDF",
                        fileSize: `${Math.floor(Math.random() * 10 + 2)}MB`,
                        pages: Math.floor(Math.random() * 200 + 50)
                    },
                    image: {
                        width: 2000,
                        height: 1500,
                        thumbnail_url: `https://picsum.photos/seed/doc${i + 1}/400/300`
                    }
                }));

                resolve({
                    results,
                    pagination: {
                        offset: offset,
                        limit: limit,
                        total: 100,
                        has_more: true
                    },
                    searchInfo: {
                        mode: mode,
                        query: query
                    }
                });
            }, 500);
        });
    }
} 