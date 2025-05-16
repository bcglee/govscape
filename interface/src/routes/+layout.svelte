<script>
  import { onMount } from 'svelte';
  
  let scrolled = false;

  onMount(() => {
    const handleScroll = () => {
      scrolled = window.scrollY > 90;
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  });
</script>

<div class="app">
  <div class="top-nav" class:scrolled>
    <nav>
      <a href="/about">About</a>
      <a href="/faq">FAQ</a>
    </nav>
  </div>

  <header class:scrolled>
    <div class="header-content">
      <a href="/" class="logo">
        <img src="/logo.png" alt="GovScape Logo" class="logo-image" />
      </a>
      <nav>
        <a href="/about">About</a>
        <a href="/faq">FAQ</a>
      </nav>
    </div>
  </header>

  <slot />
</div>

<style>
  .top-nav {
    position: fixed;
    top: 0;
    right: 1rem;
    padding: 1rem;
    z-index: 101;
    transition: opacity 0.3s ease, transform 0.3s ease;
  }

  .top-nav nav {
    display: flex;
    gap: 1rem;
  }

  .top-nav a {
    color: var(--text-color);
    text-decoration: none;
    font-weight: 600;
    padding: 0.5rem 1rem;
  }

  .top-nav.scrolled {
    opacity: 0;
    transform: translateY(-20px);
    pointer-events: none;
  }

  header {
    height: 0;
    overflow: hidden;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
    background: white;
    border-bottom: 1px solid #e9ecef;
    transition: height 0.3s ease, opacity 0.3s ease;
  }

  header.scrolled {
    height: var(--header-height);
    opacity: 1;
  }

  .header-content {
    max-width: 1400px;
    height: 100%;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .logo {
    color: var(--primary-color);
    font-family: 'Libre Baskerville', serif;
    font-size: 1.5rem;
    font-weight: bold;
    text-decoration: none;
  }

  .logo-image {
    height: 40px;
    width: auto;
    vertical-align: middle;
  }

  nav {
    display: flex;
    gap: 2rem;
  }

  nav a {
    color: var(--text-color);
    text-decoration: none;
    font-weight: 600;
  }
</style>
