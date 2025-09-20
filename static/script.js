// Global JavaScript for InhaleAI

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling for anchor links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add ripple effect to buttons
    const buttons = document.querySelectorAll('.btn, .cta-button');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = this.querySelector('.button-ripple');
            if (ripple) {
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';
                ripple.style.width = size + 'px';
                ripple.style.height = size + 'px';
                
                ripple.style.animation = 'none';
                ripple.offsetHeight; // Trigger reflow
                ripple.style.animation = 'ripple 0.6s ease-out';
            }
        });
    });

    // Add CSS for ripple animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // Add loading states for buttons
    const formButtons = document.querySelectorAll('button[type="submit"], .btn');
    formButtons.forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled) {
                const originalText = this.innerHTML;
                this.innerHTML = '<span class="btn-icon">⏳</span> Loading...';
                this.disabled = true;
                
                // Re-enable after 2 seconds (adjust as needed)
                setTimeout(() => {
                    this.innerHTML = originalText;
                    this.disabled = false;
                }, 2000);
            }
        });
    });

    // Add fade-in animation for elements
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.benefit-card, .feature-item, .step');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    // Add breathing animation to hero section
    const breathingCircles = document.querySelectorAll('.circle');
    if (breathingCircles.length > 0) {
        let breathPhase = 0;
        setInterval(() => {
            breathPhase += 0.02;
            breathingCircles.forEach((circle, index) => {
                const delay = index * 0.5;
                const scale = 1 + Math.sin(breathPhase + delay) * 0.2;
                const opacity = 0.3 + Math.sin(breathPhase + delay) * 0.5;
                circle.style.transform = `translate(-50%, -50%) scale(${scale})`;
                circle.style.opacity = opacity;
            });
        }, 50);
    }

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Escape key to stop camera (if on guide page)
        if (e.key === 'Escape') {
            const stopBtn = document.getElementById('stop-camera-btn');
            if (stopBtn && !stopBtn.disabled) {
                stopBtn.click();
            }
        }
        
        // Space key to start/stop breathing (if on guide page)
        if (e.key === ' ') {
            e.preventDefault();
            const startBreathingBtn = document.getElementById('start-breathing-btn');
            if (startBreathingBtn && !startBreathingBtn.disabled) {
                startBreathingBtn.click();
            }
        }
    });

    // Add tooltips for status indicators
    const statusItems = document.querySelectorAll('.status-item');
    statusItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = this.getAttribute('data-tooltip') || 'Status indicator';
            tooltip.style.cssText = `
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 0.8rem;
                z-index: 1000;
                pointer-events: none;
                white-space: nowrap;
            `;
            document.body.appendChild(tooltip);
            
            const rect = this.getBoundingClientRect();
            tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
            
            this.tooltip = tooltip;
        });
        
        item.addEventListener('mouseleave', function() {
            if (this.tooltip) {
                this.tooltip.remove();
                this.tooltip = null;
            }
        });
    });

    // Add data attributes for tooltips
    const cameraStatus = document.getElementById('camera-status');
    const breathingStatus = document.getElementById('breathing-status');
    
    if (cameraStatus) {
        cameraStatus.setAttribute('data-tooltip', 'Camera feed status');
    }
    if (breathingStatus) {
        breathingStatus.setAttribute('data-tooltip', 'Breathing detection status');
    }

    // Add error handling for fetch requests
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        return originalFetch.apply(this, args)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response;
            })
            .catch(error => {
                console.error('Fetch error:', error);
                // Show user-friendly error message
                showNotification('An error occurred. Please try again.', 'error');
                throw error;
            });
    };

    // Notification system
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#4ade80' : '#3b82f6'};
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            z-index: 10000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    // Add performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
            console.log(`Page loaded in ${loadTime}ms`);
        });
    }

    // Add accessibility improvements
    const focusableElements = document.querySelectorAll('button, a, input, select, textarea');
    focusableElements.forEach(element => {
        element.addEventListener('focus', function() {
            this.style.outline = '2px solid #4ade80';
            this.style.outlineOffset = '2px';
        });
        
        element.addEventListener('blur', function() {
            this.style.outline = 'none';
        });
    });

    // Add reduced motion support
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const animatedElements = document.querySelectorAll('.circle, .benefit-card, .feature-item');
        animatedElements.forEach(el => {
            el.style.animation = 'none';
            el.style.transition = 'none';
        });
    }
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Export functions for use in other scripts
window.InhaleAI = {
    debounce,
    throttle,
    showNotification: function(message, type) {
        // This will be available globally
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#4ade80' : '#3b82f6'};
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            z-index: 10000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
};
