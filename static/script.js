document.addEventListener("DOMContentLoaded", function() {
    const titleInput = document.getElementById('title');
    const suggestions = document.getElementById('suggestions');
    const articlesList = document.getElementById('articlesList');
    const pagination = document.getElementById('pagination');

    titleInput.addEventListener('input', fetchSuggestions);
    suggestions.addEventListener('click', function(e) {
        if (e.target.tagName === 'LI') {
            titleInput.value = e.target.innerText;
            suggestions.innerHTML = '';
        }
    });

    function fetchSuggestions() {
        const query = titleInput.value;
        if (query.length < 1) {
            suggestions.innerHTML = '';
            return;
        }
        
        fetch(`/autocomplete?q=${query}`)
            .then(response => response.json())
            .then(data => {
                suggestions.innerHTML = '';
                data.slice(0, 3).forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = item;
                    suggestions.appendChild(li);
                });
            });
    }

    document.getElementById('searchTitle').addEventListener('input', () => searchArticles(1));
    document.getElementById('searchKeyword').addEventListener('input', () => searchArticles(1));
    document.getElementById('searchDate').addEventListener('input', () => searchArticles(1));

    function searchArticles(page) {
        const title = document.getElementById('searchTitle').value;
        const keyword = document.getElementById('searchKeyword').value;
        const date = document.getElementById('searchDate').value;

        fetch(`/search_articles?title=${title}&keyword=${keyword}&date=${date}&page=${page}`)
            .then(response => response.json())
            .then(data => {
                articlesList.innerHTML = '';
                data.articles.forEach(article => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><a href="${article.link}" target="_blank">${article.title}</a></td>
                        <td>${article.summary}</td>
                        <td>${article.date}</td>
                    `;
                    articlesList.appendChild(row);
                });
                renderPagination(data.current_page, data.total_pages);
            });
    }

    function renderPagination(currentPage, totalPages) {
        pagination.innerHTML = '';
        if (totalPages <= 1) return;

        const createPageButton = (page, label = page) => {
            const button = document.createElement('button');
            button.textContent = label;
            button.className = 'btn btn-light';
            button.disabled = page === currentPage;
            button.addEventListener('click', () => searchArticles(page));
            return button;
        };

        if (currentPage > 1) {
            pagination.appendChild(createPageButton(currentPage - 1, 'Previous'));
        }

        const maxPagesToShow = 5;
        const ellipsis = '...';
        let startPage = Math.max(currentPage - Math.floor(maxPagesToShow / 2), 1);
        let endPage = Math.min(startPage + maxPagesToShow - 1, totalPages);

        if (endPage - startPage < maxPagesToShow) {
            startPage = Math.max(endPage - maxPagesToShow + 1, 1);
        }

        if (startPage > 1) {
            pagination.appendChild(createPageButton(1));
            if (startPage > 2) {
                pagination.appendChild(document.createTextNode(ellipsis));
            }
        }

        for (let page = startPage; page <= endPage; page++) {
            pagination.appendChild(createPageButton(page));
        }

        if (endPage < totalPages) {
            if (endPage < totalPages - 1) {
                pagination.appendChild(document.createTextNode(ellipsis));
            }
            pagination.appendChild(createPageButton(totalPages));
        }

        if (currentPage < totalPages) {
            pagination.appendChild(createPageButton(currentPage + 1, 'Next'));
        }
    }

    searchArticles(1);
});
