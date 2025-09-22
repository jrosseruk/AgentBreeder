// Theme management
class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        // Set initial theme
        this.setTheme(this.theme);

        // Add event listener to theme toggle button
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }
    }

    setTheme(theme) {
        this.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);

        // Update theme toggle button aria-label
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.setAttribute('aria-label',
                theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'
            );
            // Update icon visibility explicitly to avoid any flash
            const sun = themeToggle.querySelector('.sun-icon');
            const moon = themeToggle.querySelector('.moon-icon');
            if (sun && moon) {
                if (theme === 'dark') {
                    sun.style.display = 'block';
                    moon.style.display = 'none';
                } else {
                    sun.style.display = 'none';
                    moon.style.display = 'block';
                }
            }
        }
    }

    toggleTheme() {
        const newTheme = this.theme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }
}



// Smooth scrolling for anchor links
class SmoothScrollManager {
    constructor() {
        this.init();
    }

    init() {
        // Add smooth scrolling to all anchor links
        document.addEventListener('click', (e) => {
            const target = e.target.closest('a[href^="#"]');
            if (target) {
                e.preventDefault();
                const targetId = target.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);

                if (targetElement) {
                    const headerHeight = document.getElementById('main-header')?.offsetHeight || 0;
                    const targetPosition = targetElement.offsetTop - headerHeight - 20;

                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            }
        });
    }
}

// Copy to clipboard functionality for citation
class ClipboardManager {
    constructor() {
        this.init();
    }

    init() {
        const bibtexTextarea = document.getElementById('bibtex');
        if (bibtexTextarea) {
            bibtexTextarea.addEventListener('click', () => {
                bibtexTextarea.select();
                navigator.clipboard.writeText(bibtexTextarea.value).then(() => {
                    // Visual feedback
                    const originalBorder = bibtexTextarea.style.border;
                    bibtexTextarea.style.border = '2px solid var(--color-accent)';
                    setTimeout(() => {
                        bibtexTextarea.style.border = originalBorder;
                    }, 1000);
                }).catch(err => {
                    console.error('Failed to copy to clipboard:', err);
                });
            });
        }
    }
}

// Initialize all managers when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ThemeManager();
    new SmoothScrollManager();
    new ClipboardManager();
});

// Handle page visibility changes to re-process MathJax if needed
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.MathJax) {
        MathJax.typesetPromise().catch((err) => {
            console.warn('MathJax re-typeset failed:', err);
        });
    }
});