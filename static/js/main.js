document.addEventListener('DOMContentLoaded', function() {
    // Handle nested dropdowns
    $('.dropdown-submenu a.dropdown-toggle').on('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).next('.dropdown-menu').toggle();
    });
});
