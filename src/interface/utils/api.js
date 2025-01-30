class API {
    static async searchCollections(query, offset = 0, limit = 20, mode = 'keyword') {
        // Mock API response
        return new Promise(resolve => {
            setTimeout(() => {
                // Default results for initial page load
                const defaultResults = [
                    {
                        title: "Federal Guidelines for Cybersecurity Implementation",
                        creator: "Department of Homeland Security",
                        abstract: "Comprehensive guidelines for federal agencies on implementing cybersecurity measures and protecting critical infrastructure..."
                    },
                    {
                        title: "National Infrastructure Development Plan",
                        creator: "Department of Transportation",
                        abstract: "Strategic planning document outlining infrastructure development priorities and investment strategies for the next decade..."
                    },
                    {
                        title: "Annual Report on Environmental Protection Measures",
                        creator: "Environmental Protection Agency",
                        abstract: "Detailed analysis of environmental protection initiatives and their impact on air and water quality across the nation..."
                    },
                    {
                        title: "Public Health Policy Framework 2023",
                        creator: "Department of Health and Human Services",
                        abstract: "Comprehensive framework for public health policies, including pandemic preparedness and healthcare accessibility..."
                    },
                    {
                        title: "Digital Government Transformation Guide",
                        creator: "Office of Management and Budget",
                        abstract: "Strategic guide for modernizing government digital services and improving citizen experience through technology..."
                    },
                    {
                        title: "Federal Emergency Response Protocol",
                        creator: "Federal Emergency Management Agency",
                        abstract: "Updated protocols for coordinating federal emergency response efforts and disaster management procedures..."
                    },
                    {
                        title: "Renewable Energy Initiative Report",
                        creator: "Department of Energy",
                        abstract: "Comprehensive report on federal renewable energy initiatives and progress towards sustainability goals..."
                    },
                    {
                        title: "National Security Strategy Update",
                        creator: "Department of Defense",
                        abstract: "Strategic assessment of national security challenges and response strategies in the modern global context..."
                    },
                    {
                        title: "Federal Data Privacy Standards",
                        creator: "Federal Trade Commission",
                        abstract: "Updated standards for protecting citizen data privacy and ensuring compliance with federal regulations..."
                    },
                    {
                        title: "Economic Impact Analysis Report",
                        creator: "Department of Treasury",
                        abstract: "Analysis of economic trends and their impact on federal policy decisions and financial regulations..."
                    },
                    {
                        title: "Healthcare System Reform Proposal",
                        creator: "Centers for Medicare & Medicaid Services",
                        abstract: "Proposed reforms to improve healthcare system efficiency and accessibility for all Americans..."
                    },
                    {
                        title: "Urban Development Guidelines",
                        creator: "Department of Housing and Urban Development",
                        abstract: "Guidelines for sustainable urban development and community planning in metropolitan areas..."
                    }
                ];
                
                // Seattle-specific results for search
                const seattleResults = [
                    {
                        title: "Seattle Urban Development Report 2023",
                        creator: "Department of Housing and Urban Development",
                        abstract: "Comprehensive analysis of urban development in Seattle metropolitan area..."
                    },
                    {
                        title: "Port of Seattle Maritime Operations Guide",
                        creator: "U.S. Coast Guard",
                        abstract: "Guidelines for maritime operations and safety procedures in Seattle ports..."
                    },
                    {
                        title: "Seattle Region Transportation Infrastructure Assessment",
                        creator: "Department of Transportation",
                        abstract: "Detailed evaluation of transportation infrastructure in greater Seattle area..."
                    },
                    {
                        title: "Environmental Impact Study: Seattle Green Spaces",
                        creator: "Environmental Protection Agency",
                        abstract: "Analysis of environmental impact and preservation of Seattle's urban parks..."
                    },
                    {
                        title: "Seattle Housing Market Analysis 2023",
                        creator: "Federal Housing Administration",
                        abstract: "Current state and future projections for Seattle's housing market..."
                    }
                ];
                
                // Select appropriate results based on query
                const sourceResults = query.toLowerCase().includes('seattle') ? seattleResults : defaultResults;
                
                const results = sourceResults.map((baseResult, i) => ({
                    id: `doc-${i + 1}`,
                    url: `https://example.gov/documents/${i + 1}.pdf`,
                    title: baseResult.title,
                    metadata: {
                        text: "Official government document outlining policies, guidelines, and strategic plans",
                        date: `June 2023`,
                        creator: baseResult.creator,
                        fileType: "PDF",
                        fileSize: `${Math.floor(Math.random() * 10 + 2)}MB`,
                        pages: Math.floor(Math.random() * 200 + 50),
                        matchedPage: Math.floor(Math.random() * 200 + 1),
                        abstract: baseResult.abstract,
                        keywords: ["policy", "government", "guidelines", "implementation", "strategy"],
                        department: "Federal Government",
                        lastModified: "2023-06-15",
                        pageImages: Array.from({ length: 5 }, (_, j) => ({
                            pageNumber: j - 2,
                            thumbnailUrl: `https://picsum.photos/seed/${query ? 'seattle' : 'default'}-${i}-${j}/800/1000`
                        }))
                    },
                    image: {
                        width: 2000,
                        height: 1500,
                        thumbnail_url: `https://picsum.photos/seed/${query ? 'seattle' : 'default'}-doc-${i}/400/300`
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