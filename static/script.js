document.addEventListener("DOMContentLoaded", function() {
    const titleInput = document.getElementById('title');
    const suggestions = document.getElementById('suggestions');

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
});
