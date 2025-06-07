document.addEventListener('DOMContentLoaded', function() {
    // Generic pagination and sorting for tables with class 'paginated-sortable-table'
    document.querySelectorAll('.paginated-sortable-table').forEach(function(table) {
        const rowsPerPage = parseInt(table.dataset.rowsPerPage) || 10;
        const paginationId = table.dataset.paginationId;
        const startIndexId = table.dataset.startIndexId;
        const endIndexId = table.dataset.endIndexId;
        const totalItemsId = table.dataset.totalItemsId;
        let currentPage = 1;
        let currentSort = { column: null, asc: true };

        function getTableRows() {
            return Array.from(table.querySelectorAll('tbody tr'));
        }
        function getColumnIndex(columnName) {
            const ths = table.querySelectorAll('thead th');
            for (let i = 0; i < ths.length; i++) {
                if (ths[i].dataset.sort === columnName) return i;
            }
            return -1;
        }
        function sortRows(rows, column, asc) {
            const colIdx = getColumnIndex(column);
            if (colIdx === -1) return rows;
            return rows.slice().sort((a, b) => {
                let aVal = a.children[colIdx].getAttribute('data-value');
                let bVal = b.children[colIdx].getAttribute('data-value');
                if (aVal === null) aVal = a.children[colIdx].textContent.trim();
                if (bVal === null) bVal = b.children[colIdx].textContent.trim();
                let aNum = parseFloat(aVal.replace(/[^\d.-]/g, ''));
                let bNum = parseFloat(bVal.replace(/[^\d.-]/g, ''));
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return asc ? aNum - bNum : bNum - aNum;
                } else {
                    return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }
            });
        }
        function renderTable() {
            let rows = getTableRows();
            if (currentSort.column) {
                rows = sortRows(rows, currentSort.column, currentSort.asc);
            }
            const totalRows = rows.length;
            const totalPages = Math.ceil(totalRows / rowsPerPage);
            if (currentPage > totalPages) currentPage = totalPages || 1;
            const startIdx = (currentPage - 1) * rowsPerPage;
            const endIdx = Math.min(startIdx + rowsPerPage, totalRows);
            rows.forEach(row => row.style.display = 'none');
            rows.slice(startIdx, endIdx).forEach(row => row.style.display = '');
            renderPagination(totalPages);
            if (startIndexId) document.getElementById(startIndexId).textContent = totalRows === 0 ? 0 : startIdx + 1;
            if (endIndexId) document.getElementById(endIndexId).textContent = endIdx;
            if (totalItemsId) document.getElementById(totalItemsId).textContent = totalRows;
        }
        function renderPagination(totalPages) {
            if (!paginationId) return;
            const pagination = document.getElementById(paginationId);
            if (!pagination) return;
            pagination.innerHTML = '';
            if (totalPages <= 1) return;
            for (let i = 1; i <= totalPages; i++) {
                const li = document.createElement('li');
                li.className = 'page-item' + (i === currentPage ? ' active' : '');
                const a = document.createElement('a');
                a.className = 'page-link';
                a.href = '#';
                a.textContent = i;
                a.addEventListener('click', function(e) {
                    e.preventDefault();
                    currentPage = i;
                    renderTable();
                });
                li.appendChild(a);
                pagination.appendChild(li);
            }
        }
        table.querySelectorAll('th.sortable').forEach(th => {
            th.style.cursor = 'pointer';
            th.addEventListener('click', function() {
                const col = th.dataset.sort;
                if (currentSort.column === col) {
                    currentSort.asc = !currentSort.asc;
                } else {
                    currentSort.column = col;
                    currentSort.asc = true;
                }
                currentPage = 1;
                renderTable();
                table.querySelectorAll('th .sort-icon').forEach(icon => {
                    icon.textContent = '↕';
                });
                th.querySelector('.sort-icon').textContent = currentSort.asc ? '↑' : '↓';
            });
        });
        renderTable();
    });

    // Handle nested dropdowns (jQuery version removed for consistency)
    document.querySelectorAll('.dropdown-submenu a.dropdown-toggle').forEach(function(toggle) {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const nextMenu = this.nextElementSibling;
            if (nextMenu && nextMenu.classList.contains('dropdown-menu')) {
                nextMenu.classList.toggle('show');
            }
        });
    });
});
