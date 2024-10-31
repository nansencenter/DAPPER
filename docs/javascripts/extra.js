// https://squidfunk.github.io/mkdocs-material/customization/#additional-javascript
console.log("This message appears on every page")

// Tag converted notebooks (to facilitate custom styling)
document.addEventListener("DOMContentLoaded", function() {
  if (document.querySelector('.jp-Notebook')) {
    document.body.classList.add('document-is-notebook');
  }
});

// Hide "Home" h1
document.querySelectorAll('h1').forEach(function(h1) {
    if (h1.textContent.trim() === "Home") {
        h1.style.display = 'none';
    }
});
