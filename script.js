// Initialize Lucide icons
lucide.createIcons();

// Navigation scroll effect
const nav = document.getElementById('nav');
let lastScrollY = window.scrollY;

window.addEventListener('scroll', () => {
  if (window.scrollY > 50) {
    nav.classList.add('nav-scrolled');
  } else {
    nav.classList.remove('nav-scrolled');
  }
  lastScrollY = window.scrollY;
});

// Mobile navigation toggle
const navToggle = document.querySelector('.nav-toggle');
const mobileNav = document.getElementById('mobile-nav');

function closeMobileNav() {
  mobileNav.classList.remove('open');
  navToggle.setAttribute('aria-expanded', 'false');
  navToggle.querySelector('i').setAttribute('data-lucide', 'menu');
  lucide.createIcons();
}

function openMobileNav() {
  mobileNav.classList.add('open');
  navToggle.setAttribute('aria-expanded', 'true');
  navToggle.querySelector('i').setAttribute('data-lucide', 'x');
  lucide.createIcons();
}

navToggle.setAttribute('aria-expanded', 'false');
navToggle.setAttribute('aria-controls', 'mobile-nav');

navToggle.addEventListener('click', () => {
  if (mobileNav.classList.contains('open')) {
    closeMobileNav();
  } else {
    openMobileNav();
  }
});

// Close mobile nav when clicking a link
document.querySelectorAll('.mobile-nav .nav-link').forEach(link => {
  link.addEventListener('click', () => closeMobileNav());
});

// Close mobile nav with Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && mobileNav.classList.contains('open')) {
    closeMobileNav();
    navToggle.focus();
  }
});

// Project filtering
const filterButtons = document.querySelectorAll('.filter-btn');
const projectCards = document.querySelectorAll('.project-card');

filterButtons.forEach(button => {
  button.addEventListener('click', () => {
    // Update active state and aria-pressed
    filterButtons.forEach(btn => {
      btn.classList.remove('active');
      btn.setAttribute('aria-pressed', 'false');
    });
    button.classList.add('active');
    button.setAttribute('aria-pressed', 'true');

    const filter = button.dataset.filter;

    projectCards.forEach(card => {
      if (filter === 'all') {
        card.classList.remove('card-hidden');
      } else {
        const tags = card.dataset.tags.split(' ');
        if (tags.includes(filter)) {
          card.classList.remove('card-hidden');
        } else {
          card.classList.add('card-hidden');
        }
      }
    });
  });
});

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      const navHeight = nav.offsetHeight;
      const targetPosition = target.offsetTop - navHeight - 20;
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    }
  });
});

// Intersection Observer for scroll animations
const observerOptions = {
  root: null,
  rootMargin: '0px',
  threshold: 0.1
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animation = 'fadeInUp 0.6s ease forwards';
      observer.unobserve(entry.target);
    }
  });
}, observerOptions);

// Observe elements for scroll animations
document.querySelectorAll('.card, .skill-category, .section-title').forEach(el => {
  el.style.opacity = '0';
  observer.observe(el);
});

// Active navigation link highlighting
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
  let current = '';
  const navHeight = nav.offsetHeight;

  sections.forEach(section => {
    const sectionTop = section.offsetTop - navHeight - 100;
    const sectionHeight = section.offsetHeight;

    if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
      current = section.getAttribute('id');
    }
  });

  navLinks.forEach(link => {
    link.classList.remove('nav-link-active');
    if (link.getAttribute('href') === `#${current}`) {
      link.classList.add('nav-link-active');
    }
  });
});
