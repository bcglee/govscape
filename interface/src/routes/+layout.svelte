<div class="app">
  <header>
    <div class="header-content">
      <a href="/" class="logo">GovScape</a>
      <nav>
        <a href="/about">About</a>
        <a href="/faq">FAQ</a>
      </nav>
    </div>
  </header>

  <slot />
</div>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    background-color: var(--background-color);
  }

  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    height: var(--header-height);
    background: white;
    border-bottom: 1px solid #e9ecef;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
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

  nav {
    display: flex;
    gap: 2rem;
  }

  nav a {
    color: var(--text-color);
    text-decoration: none;
    font-weight: 600;
  }

  nav a:hover {
    color: var(--primary-color);
  }

  main {
    margin-top: var(--header-height);
    flex: 1;
    padding: 2rem;
  }

  h1 {
    color: var(--text-color);
    font-size: 2.5rem;
    margin-bottom: 2rem;
    font-weight: bold;
  }
</style>
