"""
HTML for every page, in design A ("clinical instrument"). Styles live in
static/assets/alati.css; each page adds only what is specific to it.

Element ids are part of the contract with the inline scripts (and with the
end-to-end tests), so restyling should keep them.
"""

CONTACT = "ahmadalaty@gmail.com"
AI_RELEASE = "v1.0"   # what users see as the AI's version; /model_info keeps the internal build name
ASSET_VERSION = "20261120a"

HEAD = """<meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="preload" href="/assets/fonts/public-sans-latin.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="/assets/alati.css?v=""" + ASSET_VERSION + "\">"

TOPBAR_PUBLIC = """<header class="topbar"><div class="topbar-inner">
    <a class="brand" href="/"><span class="brand-mark" aria-hidden="true"></span>Alati <small>by Nabeeh</small></a>
    <nav class="nav" aria-label="Site">
      <a href="/pricing">Pricing</a>
      <a href="/legal/terms">Terms</a>
      <a href="/legal/privacy">Privacy</a>
      <a href="mailto:""" + CONTACT + """">Contact</a>
    </nav>
  </div></header>"""


def _topbar_app(current: str) -> str:
    def cur(name):
        return ' aria-current="page"' if name == current else ""
    return ("""<header class="topbar"><div class="topbar-inner">
    <a class="brand" href="/scan"><span class="brand-mark" aria-hidden="true"></span>Alati</a>
    <nav class="nav" aria-label="Main">
      <a href="/scan" id="scan-btn\"""" + cur("scan") + """>Scan</a>
      <a href="/account\"""" + cur("account") + """>My account</a>
      <a href="/dashboard" id="admin-btn" style="display: none;\"""" + cur("admin") + """>Admin</a>
      <button type="button" onclick="logout()">Sign out</button>
    </nav>
  </div></header>""")


FOOTER = """<footer class="footer"><div class="footer-inner">
    <a href="/legal/terms">Terms of Service</a>
    <a href="/legal/privacy">Privacy Notice</a>
    <a href="/pricing">Pricing</a>
    <a href="mailto:""" + CONTACT + """">Contact</a>
    <a href="https://nabeeh.health">Nabeeh</a>
  </div></footer>"""


def _fill(html: str, current: str = "") -> str:
    return (html.replace("{{HEAD}}", HEAD)
                .replace("{{TOPBAR_PUBLIC}}", TOPBAR_PUBLIC)
                .replace("{{TOPBAR_APP}}", _topbar_app(current))
                .replace("{{FOOTER}}", FOOTER)
                .replace("{{CONTACT}}", CONTACT)
                .replace("{{AI_RELEASE}}", AI_RELEASE))


# =====================================================================
# Home / sign-in
# =====================================================================
LOGIN_HTML = _fill(r"""<!DOCTYPE html>
<html lang="en">
<head>
  {{HEAD}}
  <title>Alati - Diabetic retinopathy screening</title>
  <style>
    .home { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); grid-template-areas: "intro auth" "details auth";
            column-gap: 44px; row-gap: 0; align-items: start; padding-top: 44px; }
    .home-intro { grid-area: intro; }
    .home-details { grid-area: details; }
    .auth { grid-area: auth; }
    .hero-title { font-size: 32px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; }
    .lead { color: var(--ink-2); font-size: 16px; margin-top: 12px; max-width: 48ch; }
    table.figs { border-collapse: collapse; margin-top: 26px; width: 100%; max-width: 480px; font-size: 14px; }
    table.figs th, table.figs td { text-align: left; padding: 9px 12px 9px 0; border-bottom: 1px solid var(--line); }
    table.figs th { font-weight: 500; color: var(--muted-2); font-size: 12px; }
    table.figs td.n { font-family: var(--mono); font-size: 14px; white-space: nowrap; }
    table.figs td.n small { color: var(--muted-2); font-size: 11.5px; }
    .figs-src { margin-top: 10px; max-width: 480px; line-height: 1.6; }
    .caveat { margin-top: 10px; max-width: 480px; }
    .plans-table td:first-child { font-weight: 600; white-space: nowrap; }
    .plans-table td:nth-child(2) { font-family: var(--mono); white-space: nowrap; }
    .auth { padding: 24px; }
    .tabs { display: flex; gap: 22px; border-bottom: 1px solid var(--line); margin-bottom: 20px; }
    .tab-btn { background: none; border: 0; font: inherit; font-size: 15px; color: var(--muted-2); padding: 0 0 10px; cursor: pointer; }
    .tab-btn.active { color: var(--ink); font-weight: 600; box-shadow: inset 0 -2px var(--blue); }
    .form-section { display: none; }
    .form-section.active { display: block; }
    .form-row-end { display: flex; justify-content: flex-end; margin: -6px 0 16px; font-size: 13px; }
    .switch-line { text-align: center; font-size: 13.5px; color: var(--muted); margin-top: 14px; }
    .form-intro { font-size: 14px; color: var(--ink-2); margin-bottom: 16px; }
    .error-msg { color: var(--red); font-size: 13.5px; margin-bottom: 12px; }
    .verify-banner, .success-banner { display: none; margin-bottom: 16px; }
    .verify-banner.active, .success-banner.active { display: block; }
    .info-box { font-size: 12.5px; color: var(--muted); margin: -8px 0 14px; }
    .disclaimer-box { border: 1px solid var(--line); background: var(--surface-2); border-radius: 4px; padding: 12px 14px;
                      max-height: 210px; overflow-y: auto; font-size: 12.5px; color: var(--ink-2); line-height: 1.55; margin-bottom: 14px; }
    .disclaimer-box h4 { font-size: 13px; margin-bottom: 6px; }
    .disclaimer-box p + p { margin-top: 8px; }
    .disclaimer-box ul { margin: 6px 0 0 18px; display: grid; gap: 3px; }
    .consent-row { display: flex; gap: 9px; align-items: flex-start; font-size: 13.5px; margin-bottom: 16px; }
    .consent-row label { cursor: pointer; }
    @media (max-width: 860px) {
      .home { grid-template-columns: 1fr; grid-template-areas: "intro" "auth" "details"; row-gap: 24px; padding-top: 24px; }
      .hero-title { font-size: 26px; }
      .plans-table tr { display: grid; grid-template-columns: auto 1fr; }
      table.data.plans-table td { border-bottom: 0; }
      .plans-table td:nth-child(3) { grid-column: 1 / -1; padding-top: 0; color: var(--muted); }
      .plans-table tr + tr { border-top: 1px solid var(--line-3); }
    }
  </style>
</head>
<body>
  {{TOPBAR_PUBLIC}}

  <main class="page home">
    <section class="home-intro">
      <h1 class="hero-title">Diabetic retinopathy screening from a fundus photograph</h1>
      <p class="lead">Upload a colour fundus photograph and get a referral decision in seconds. For licensed clinicians, as an aid to clinical judgement.</p>
    </section>

    <section class="home-details">

      <table class="figs" aria-label="Measured performance">
        <tr><th>Measured on</th><th>Sensitivity</th><th>Specificity</th></tr>
        <tr><td>Referable DR</td><td class="n">99.4% <small>98.9–99.7</small></td><td class="n">79.2% <small>77.3–81.0</small></td></tr>
        <tr><td>Any DR</td><td class="n">99.8% <small>99.5–99.9</small></td><td class="n">89.0% <small>87.4–90.5</small></td></tr>
      </table>
      <p class="meta figs-src">Alati {{AI_RELEASE}} · APTOS 2019 · 3,662 photographs · independent of development · 3,406 gradable · retrospective · standard fundus cameras · 95% CI</p>
      <p class="fine caveat">Low-resolution and heavily compressed photos are declined rather than called normal. Smartphone fundus cameras are not yet validated. A screening aid, not a replacement for an eye examination.</p>

      <h2 class="section-title" id="pricing">Plans</h2>
      <div class="panel table-wrap">
        <table class="data plans-table">
          <tr><td>Free</td><td>$0</td><td>5 scans per day · referable DR result per photo · image-quality check · PDF report and history</td></tr>
          <tr><td>Premium</td><td><span class="chip blue">Coming soon</span></td><td>Higher daily volume for clinics and screening programmes</td></tr>
        </table>
      </div>
      <p class="fine" style="margin-top: 8px;">Alati is free to use while Premium is in preparation. Need a higher scan volume or an evaluation account? Email <a href="mailto:{{CONTACT}}">{{CONTACT}}</a>.</p>
    </section>

    <section class="panel auth" aria-label="Sign in or create an account">
      <div class="tabs" role="tablist">
        <button type="button" class="tab-btn active" data-tab="login" onclick="switchTab('login')">Sign in</button>
        <button type="button" class="tab-btn" data-tab="register" onclick="switchTab('register')">Create account</button>
      </div>

      <form id="login" class="form-section active" onsubmit="handleLogin(event)" novalidate>
        <div id="verify-banner" class="notice warn verify-banner">
          <div class="t">Email not verified</div>
          <div id="verify-text">Please verify your email address before signing in. Check your inbox for the verification link.</div>
          <button type="button" class="linklike" style="margin-top: 6px;" onclick="resendVerification()">Resend verification email</button>
        </div>
        <div id="success-banner" class="notice ok success-banner">
          <div class="t" id="success-title">Account created</div>
          <div id="success-text">Please check your email and click the verification link to activate your account.</div>
        </div>
        <div id="login-error" class="error-msg" style="display: none;"></div>
        <div class="field">
          <label class="label" for="login-email">Email address</label>
          <input type="email" id="login-email" placeholder="doctor@clinic.com" autocomplete="email">
        </div>
        <div class="field">
          <label class="label" for="login-password">Password</label>
          <input type="password" id="login-password" placeholder="••••••••" autocomplete="current-password">
        </div>
        <div class="form-row-end"><button type="button" class="linklike" onclick="switchTab('forgot')">Forgot password?</button></div>
        <button type="submit" class="btn btn-block" id="login-btn">Sign in</button>
        <p class="switch-line">Don't have an account? <button type="button" class="linklike" onclick="switchTab('register')">Create one</button></p>
      </form>

      <form id="forgot" class="form-section" onsubmit="handleForgot(event)" novalidate>
        <p class="form-intro">Enter the email address you registered with. We'll send you a link to choose a new password.</p>
        <div id="forgot-error" class="error-msg" style="display: none;"></div>
        <div id="forgot-done" class="notice ok success-banner">
          <div class="t">Check your email</div>
          <div id="forgot-done-text">If an account exists for that address, a reset link is on its way. The link expires in 1 hour.</div>
        </div>
        <div class="field">
          <label class="label" for="forgot-email">Email address</label>
          <input type="email" id="forgot-email" placeholder="doctor@clinic.com" autocomplete="email">
        </div>
        <button type="submit" class="btn btn-block" id="forgot-btn">Send reset link</button>
        <p class="switch-line">Remembered it? <button type="button" class="linklike" onclick="switchTab('login')">Back to sign in</button></p>
      </form>

      <form id="register" class="form-section" onsubmit="handleRegister(event)" novalidate>
        <div id="register-error" class="error-msg" style="display: none;"></div>
        <div class="field">
          <label class="label" for="register-name">Full name</label>
          <input type="text" id="register-name" placeholder="Dr. Jane Smith" autocomplete="name" maxlength="120">
        </div>
        <div class="field">
          <label class="label" for="register-email">Email address</label>
          <input type="email" id="register-email" placeholder="doctor@clinic.com" autocomplete="email">
        </div>
        <div class="field">
          <label class="label" for="register-password">Password</label>
          <input type="password" id="register-password" placeholder="••••••••" autocomplete="new-password">
        </div>
        <p class="info-box">At least 8 characters.</p>
        <div class="disclaimer-box">
          <h4>Important: please read before use</h4>
          <p>This tool is an AI-based screening aid for diabetic retinopathy. It is currently in a pre-approval / research-use stage and has not yet received regulatory clearance (e.g., FDA, CE, or local Ministry of Health approval) in any jurisdiction.</p>
          <p>This tool is intended solely as a supportive aid to a licensed physician's own clinical judgment. It does not diagnose, and its output must not be used as the sole basis for any clinical decision. All findings should be independently verified by the treating physician.</p>
          <p>By creating an account and using this tool, you confirm that:</p>
          <ul>
            <li>You are a licensed medical professional using this tool within your own clinical judgment and responsibility</li>
            <li>You understand this tool has not received regulatory approval and are choosing to use it in its current pre-approval state</li>
            <li>You will not rely on this tool as a sole or final diagnostic determination</li>
            <li>You accept full responsibility for clinical decisions made in your practice, including any use of this tool's output</li>
          </ul>
        </div>
        <div class="consent-row">
          <input type="checkbox" id="register-consent">
          <label for="register-consent">I have read and understood the above, and I agree to these terms.</label>
        </div>
        <button type="submit" class="btn btn-block" id="register-btn">Create account</button>
        <p class="switch-line">Already have an account? <button type="button" class="linklike" onclick="switchTab('login')">Sign in</button></p>
      </form>
    </section>
  </main>

  {{FOOTER}}

  <script>
    function switchTab(tab) {
      document.querySelectorAll('.form-section').forEach(el => el.classList.remove('active'));
      document.getElementById(tab).classList.add('active');
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.toggle('active', el.dataset.tab === tab));
      if (tab === 'forgot') {
        document.getElementById('forgot-email').value = document.getElementById('login-email').value;
        document.getElementById('forgot-done').classList.remove('active');
        document.getElementById('forgot-error').style.display = 'none';
      }
    }

    let lastEmailForResend = '';

    function hideAllBanners() {
      document.getElementById('verify-banner').classList.remove('active');
      document.getElementById('success-banner').classList.remove('active');
      document.getElementById('login-error').style.display = 'none';
    }
    function showError(id, text) { const el = document.getElementById(id); el.textContent = text; el.style.display = 'block'; }
    async function readDetail(res, fallback) {
      try { const b = await res.json(); if (typeof b.detail === 'string') return b.detail; return b.message || fallback; }
      catch (e) { return fallback; }
    }
    function setBusy(id, busy, label) { const b = document.getElementById(id); b.disabled = busy; if (label) b.textContent = label; }

    async function handleLogin(ev) {
      if (ev) ev.preventDefault();
      const email = document.getElementById('login-email').value.trim();
      const password = document.getElementById('login-password').value;
      hideAllBanners();
      lastEmailForResend = email;
      if (!email || !password) { showError('login-error', 'Please enter your email and password.'); return; }
      setBusy('login-btn', true, 'Signing in…');
      try {
        const res = await fetch('/auth/login', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({email, password}) });
        if (res.ok) {
          const data = await res.json();
          localStorage.setItem('token', data.access_token);
          localStorage.setItem('is_admin', data.is_admin ? 'true' : 'false');
          window.location.href = data.is_admin === true ? '/dashboard' : '/scan';
          return;
        }
        const detail = await readDetail(res, 'Incorrect email or password.');
        if (res.status === 403 && /verif/i.test(detail)) {
          document.getElementById('verify-text').textContent = detail;
          document.getElementById('verify-banner').classList.add('active');
        } else if (res.status === 401) {
          showError('login-error', 'Incorrect email or password.');
        } else {
          showError('login-error', detail);
        }
      } catch (err) {
        showError('login-error', 'Could not reach the server. Check your connection and try again.');
      }
      setBusy('login-btn', false, 'Sign in');
    }

    async function handleRegister(ev) {
      if (ev) ev.preventDefault();
      const full_name = document.getElementById('register-name').value.trim();
      const email = document.getElementById('register-email').value.trim();
      const password = document.getElementById('register-password').value;
      const consent = document.getElementById('register-consent').checked;
      document.getElementById('register-error').style.display = 'none';
      hideAllBanners();
      if (!full_name || !email || !password) { showError('register-error', 'Please fill in your name, email and password.'); return; }
      if (password.length < 8) { showError('register-error', 'Password must be at least 8 characters.'); return; }
      if (!consent) { showError('register-error', 'Please read and agree to the terms above before creating an account.'); return; }
      setBusy('register-btn', true, 'Creating account…');
      try {
        const res = await fetch('/auth/register', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({email, password, full_name}) });
        if (res.ok) {
          const data = await res.json();
          if (data.status === 'verification_sent') {
            switchTab('login');
            document.getElementById('login-email').value = email;
            document.getElementById('success-title').textContent = 'Account created';
            document.getElementById('success-text').textContent = 'We sent a verification link to ' + email + '. Open it to activate your account, then sign in here. If it is not in your inbox, check the spam folder.';
            document.getElementById('success-banner').classList.add('active');
            lastEmailForResend = email;
          } else if (data.access_token) {
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('is_admin', data.is_admin ? 'true' : 'false');
            window.location.href = '/scan';
            return;
          }
        } else {
          showError('register-error', await readDetail(res, 'Registration failed. Please try again.'));
        }
      } catch (err) {
        showError('register-error', 'Could not reach the server. Check your connection and try again.');
      }
      setBusy('register-btn', false, 'Create account');
    }

    async function handleForgot(ev) {
      if (ev) ev.preventDefault();
      const email = document.getElementById('forgot-email').value.trim();
      document.getElementById('forgot-error').style.display = 'none';
      document.getElementById('forgot-done').classList.remove('active');
      if (!email) { showError('forgot-error', 'Please enter your email address.'); return; }
      setBusy('forgot-btn', true, 'Sending…');
      try {
        const res = await fetch('/auth/forgot-password', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({email}) });
        if (res.ok) {
          const data = await res.json();
          document.getElementById('forgot-done-text').textContent = data.message;
          document.getElementById('forgot-done').classList.add('active');
        } else {
          showError('forgot-error', await readDetail(res, 'Could not send the reset link. Please try again.'));
        }
      } catch (e) {
        showError('forgot-error', 'Could not reach the server. Check your connection and try again.');
      }
      setBusy('forgot-btn', false, 'Send reset link');
    }

    async function resendVerification() {
      const email = lastEmailForResend || document.getElementById('login-email').value.trim();
      const text = document.getElementById('verify-text');
      if (!email) { text.textContent = 'Enter your email address above first.'; return; }
      try {
        const res = await fetch('/auth/resend-verification', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({email}) });
        const data = await res.json();
        text.textContent = data.message || data.detail || 'Check your email for the verification link.';
      } catch (e) {
        text.textContent = 'Could not resend the verification email. Please try again.';
      }
    }

    if (localStorage.getItem('token')) {
      fetch('/user/me', {headers: {'Authorization': 'Bearer ' + localStorage.getItem('token')}})
        .then(r => { if (r.ok) window.location.href = localStorage.getItem('is_admin') === 'true' ? '/dashboard' : '/scan'; })
        .catch(() => {});
    }
  </script>
</body>
</html>""")


# =====================================================================
# Scan
# =====================================================================
SCAN_HTML = _fill(r"""<!DOCTYPE html>
<html lang="en">
<head>
  {{HEAD}}
  <title>Alati - New scan</title>
  <style>
    .seg { display: inline-flex; border: 1px solid var(--line-2); border-radius: 4px; overflow: hidden; background: var(--surface); }
    .seg input { position: absolute; opacity: 0; pointer-events: none; }
    .seg label { padding: 8px 16px; font-size: 14px; color: var(--ink-2); cursor: pointer; border-right: 1px solid var(--line-2); user-select: none; }
    .seg label:last-of-type { border-right: 0; }
    .seg input:checked + label { background: var(--blue); color: #fff; font-weight: 600; }
    .seg input:focus-visible + label { outline: 2px solid var(--blue); outline-offset: -2px; }
    .uploads { display: grid; grid-template-columns: 1fr; gap: 14px; }
    .uploads.both { grid-template-columns: 1fr 1fr; }
    .upload-area { position: relative; border: 1px dashed var(--line-2); border-radius: 6px; background: var(--surface); min-height: 190px;
                   display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; padding: 22px 16px; text-align: center; cursor: pointer; }
    .upload-area:hover, .upload-area.drag { border-color: var(--blue); background: #fbfcfe; }
    .upload-area.has-file { border-style: solid; border-color: var(--line-2); }
    .upload-eye { font-family: var(--mono); font-size: 12px; color: var(--muted-2); letter-spacing: 0.04em; }
    .upload-icon { width: 30px; height: 30px; color: var(--muted-2); }
    .upload-text { font-size: 14px; color: var(--ink-2); }
    .upload-hint { font-size: 12.5px; color: var(--muted-2); }
    .upload-preview { width: 116px; height: 116px; border-radius: 50%; object-fit: cover; background: #000; display: none; }
    .upload-area.has-file .upload-preview { display: block; }
    .upload-area.has-file .upload-icon, .upload-area.has-file .upload-hint { display: none; }
    .file-name { font-family: var(--mono); font-size: 12px; color: var(--ink-2); word-break: break-all; display: none; }
    .upload-area.has-file .file-name { display: block; }
    .guidance { font-size: 13px; color: var(--muted); margin-top: 12px; max-width: 90ch; }
    .actions { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
    .btn-row { display: flex; gap: 10px; }
    #analyze-btn { min-width: 130px; }
    .processing { display: none; align-items: center; gap: 12px; margin-top: 18px; padding: 12px 14px; }
    .processing.active { display: flex; }
    .alert { display: none; margin-top: 18px; }
    .alert.active { display: block; }
    #results { display: none; }
    .result-list .rhead, .result-card { display: grid; grid-template-columns: 70px 62px minmax(0, 1fr) auto; gap: 14px; align-items: center; padding: 12px 16px; }
    .result-list .rhead { font-size: 12px; color: var(--muted-2); background: var(--surface-2); border-bottom: 1px solid var(--line); padding-block: 9px; }
    .result-card + .result-card { border-top: 1px solid var(--line-3); }
    .rc-eye { font-family: var(--mono); font-size: 12.5px; color: var(--grey); }
    .rc-photo { width: 52px; height: 52px; border-radius: 50%; object-fit: cover; background: #000; }
    .result-headline { font-weight: 600; font-size: 15px; }
    .result-detail { font-size: 13px; color: var(--muted); margin-top: 2px; }
    .result-card.red .result-headline { color: var(--red); }
    .result-card.amber .result-headline { color: var(--amber); }
    .result-card.green .result-headline { color: var(--green); }
    .result-card.grey .result-headline { color: var(--grey); }
    .result-actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
    #result-meta { margin-top: 10px; }
    .fine.disclaimer { margin-top: 14px; max-width: 90ch; }
    @media (max-width: 720px) {
      .uploads.both { grid-template-columns: 1fr; }
      .result-list .rhead { display: none; }
      .result-card { grid-template-columns: 56px minmax(0, 1fr); gap: 4px 12px; }
      .rc-photo { grid-row: span 3; width: 56px; height: 56px; }
      .rc-eye, .result-card > div:nth-child(3), .rc-action { grid-column: 2; }
      .rc-action { margin-top: 4px; }
      .actions { flex-direction: column-reverse; align-items: stretch; }
      .btn-row .btn { flex: 1; }
    }
  </style>
</head>
<body>
  {{TOPBAR_APP}}

  <main class="page">
    <div class="page-head">
      <div>
        <h1 class="title">New scan</h1>
        <p class="sub">Diabetic retinopathy screening from a colour fundus photograph</p>
      </div>
      <div class="seg" role="radiogroup" aria-label="Which eye">
        <input type="radio" id="left" name="eye" value="left" checked><label for="left">Left eye</label>
        <input type="radio" id="right" name="eye" value="right"><label for="right">Right eye</label>
        <input type="radio" id="both" name="eye" value="both"><label for="both">Both eyes</label>
      </div>
    </div>

    <div class="uploads" id="uploads">
      <div class="upload-area" id="zone-a" data-slot="a" tabindex="0" role="button" aria-label="Choose a fundus photograph">
        <div class="upload-eye" id="zone-a-eye" style="display:none;">LEFT EYE</div>
        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>
        <img class="upload-preview" alt="Selected photograph">
        <div class="upload-text">Click to choose a photo, or drag it here</div>
        <div class="upload-hint">JPG or PNG, up to 20 MB</div>
        <div class="file-name"></div>
      </div>
      <div class="upload-area" id="zone-b" data-slot="b" tabindex="0" role="button" aria-label="Choose the right-eye photograph" style="display:none;">
        <div class="upload-eye">RIGHT EYE</div>
        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>
        <img class="upload-preview" alt="Selected right-eye photograph">
        <div class="upload-text">Click to choose a photo, or drag it here</div>
        <div class="upload-hint">JPG or PNG, up to 20 MB</div>
        <div class="file-name"></div>
      </div>
    </div>
    <input type="file" id="file-a" accept="image/jpeg,image/png,image/webp,image/tiff" hidden>
    <input type="file" id="file-b" accept="image/jpeg,image/png,image/webp,image/tiff" hidden>
    <p class="guidance">Upload the original colour fundus photograph from a standard fundus camera. Screenshots, downloaded copies and photos sent through messaging apps are often too small or too compressed: Alati can still flag referable disease in them, but will not rule it out. Smartphone fundus cameras and ultra-widefield images are not yet validated. Photos are analysed in memory and are not stored.</p>

    <div class="actions">
      <span class="meta" id="usage-note"></span>
      <div class="btn-row">
        <button type="button" class="btn btn-quiet" onclick="resetForm()">Clear</button>
        <button type="button" class="btn" id="analyze-btn" onclick="submitScan()">Analyze</button>
      </div>
    </div>

    <div class="panel processing" id="processing" role="status">
      <span class="spinner dark" aria-hidden="true"></span>
      <div><div style="font-weight:600;">Analysing the photograph…</div><div class="fine">This usually takes 5–15 seconds per eye.</div></div>
    </div>

    <div class="notice error alert" id="error-box" role="alert"></div>

    <section id="results" aria-live="polite">
      <h2 class="section-title">Result</h2>
      <div class="panel result-list">
        <div class="rhead"><span>Eye</span><span>Photo</span><span>Finding</span><span>Action</span></div>
        <div id="result-grid"></div>
      </div>
      <p class="meta" id="result-meta"></p>
      <div class="result-actions">
        <button type="button" class="btn" id="report-btn" onclick="downloadReport()">Download PDF report</button>
        <button type="button" class="btn btn-ghost" id="new-scan-btn" onclick="resetForm(); window.scrollTo({top: 0, behavior: 'smooth'});">New scan</button>
      </div>
      <p class="fine disclaimer">Alati is a screening aid for licensed clinicians, not a diagnosis. Confirm every result clinically. It screens for diabetic retinopathy only and does not exclude other eye disease.</p>
    </section>
  </main>

  {{FOOTER}}

  <script>
    const AI_RELEASE = '{{AI_RELEASE}}';
    const MAX_BYTES = 20 * 1024 * 1024;
    const token = localStorage.getItem('token');
    if (!token) window.location.replace('/login');
    const files = { a: null, b: null };
    let lastScanId = null;

    if (localStorage.getItem('is_admin') === 'true') document.getElementById('admin-btn').style.display = '';

    function logout() {
      localStorage.removeItem('token');
      localStorage.removeItem('is_admin');
      window.location.href = '/login';
    }
    function escapeHtml(s) {
      if (s == null) return '';
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    // Stored result text (from inference.py) -> what a clinician should read.
    function describe(dx) {
      const s = dx || '';
      if (/not gradable/i.test(s)) {
        const reason = s.split('—').slice(1).join('—').trim();
        return { tone: 'grey', headline: 'Image not gradable', action: 'Retake or re-upload',
                 detail: (reason ? reason + ' ' : '') + 'No result was produced. This is not a normal result.' };
      }
      if (/not referable/i.test(s)) {
        return { tone: 'amber', headline: 'Diabetic retinopathy, not referable', action: 'Monitor and rescreen',
                 detail: 'Signs of diabetic retinopathy below the referral threshold. This is not a normal result.' };
      }
      if (/referable/i.test(s)) {
        return { tone: 'red', headline: 'Referable diabetic retinopathy', action: 'Refer to ophthalmology',
                 detail: 'Ophthalmology review recommended. Severity is not graded; assess the eye clinically.' };
      }
      if (/^normal$/i.test(s.trim())) {
        return { tone: 'green', headline: 'No diabetic retinopathy detected', action: 'Rescreen as usual',
                 detail: 'Rescreen at the usual interval. This does not exclude other eye disease.' };
      }
      return { tone: 'grey', headline: s || 'No result', action: '', detail: '' };
    }

    function selectedMode() { return document.querySelector('input[name="eye"]:checked').value; }

    function applyMode() {
      const both = selectedMode() === 'both';
      document.getElementById('uploads').classList.toggle('both', both);
      document.getElementById('zone-b').style.display = both ? '' : 'none';
      document.getElementById('zone-a-eye').style.display = both ? '' : 'none';
      document.getElementById('zone-a').setAttribute('aria-label', both ? 'Choose the left-eye photograph' : 'Choose a fundus photograph');
      if (!both) clearSlot('b');
    }
    document.querySelectorAll('input[name="eye"]').forEach(r => r.addEventListener('change', applyMode));

    function showError(text) { const b = document.getElementById('error-box'); b.textContent = text; b.classList.add('active'); }
    function hideError() { document.getElementById('error-box').classList.remove('active'); }

    function setSlot(slot, file) {
      hideError();
      if (!file) return;
      if (!/^image\//.test(file.type) && !/\.(jpe?g|png|tiff?|webp)$/i.test(file.name)) {
        showError('"' + file.name + '" is not an image. Choose a JPG or PNG fundus photograph.');
        return;
      }
      if (file.size > MAX_BYTES) { showError('"' + file.name + '" is larger than 20 MB. Choose a smaller file.'); return; }
      files[slot] = file;
      const zone = document.getElementById('zone-' + slot);
      const img = zone.querySelector('.upload-preview');
      if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
      img.dataset.url = URL.createObjectURL(file);
      img.src = img.dataset.url;
      zone.classList.add('has-file');
      zone.querySelector('.upload-text').textContent = 'Click to change the photo';
      zone.querySelector('.file-name').textContent = file.name + ' · ' + (file.size / 1024 / 1024).toFixed(1) + ' MB';
    }

    function clearSlot(slot) {
      files[slot] = null;
      const zone = document.getElementById('zone-' + slot);
      const img = zone.querySelector('.upload-preview');
      if (img.dataset.url) { URL.revokeObjectURL(img.dataset.url); delete img.dataset.url; }
      img.removeAttribute('src');
      zone.classList.remove('has-file');
      zone.querySelector('.upload-text').textContent = 'Click to choose a photo, or drag it here';
      zone.querySelector('.file-name').textContent = '';
      document.getElementById('file-' + slot).value = '';
    }

    ['a', 'b'].forEach(slot => {
      const zone = document.getElementById('zone-' + slot);
      const input = document.getElementById('file-' + slot);
      zone.addEventListener('click', () => input.click());
      zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
      input.addEventListener('change', e => setSlot(slot, e.target.files[0]));
      zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
      zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
      zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag'); if (e.dataTransfer.files.length) setSlot(slot, e.dataTransfer.files[0]); });
    });

    async function loadUsage() {
      try {
        const res = await fetch('/user/me', { headers: { 'Authorization': 'Bearer ' + token } });
        if (res.status === 401) { logout(); return; }
        if (!res.ok) return;
        const u = await res.json();
        const note = document.getElementById('usage-note');
        if (u.is_banned) {
          note.textContent = 'Account suspended';
          document.getElementById('analyze-btn').disabled = true;
          showError('Your account is suspended, so new scans are disabled. Contact {{CONTACT}} for help.');
        } else if (u.usage_limit === -1 || u.usage_limit === null) {
          note.textContent = 'Unlimited scans';
        } else {
          const left = Math.max(0, u.usage_limit - u.usage_count);
          note.textContent = left + ' of ' + u.usage_limit + ' scans left today';
        }
      } catch (e) {}
    }

    async function submitScan() {
      const mode = selectedMode();
      hideError();
      if (mode === 'both' && (!files.a || !files.b)) { showError('Choose a photograph for each eye, or select a single eye.'); return; }
      if (mode !== 'both' && !files.a) { showError('Choose a fundus photograph first.'); return; }

      const formData = new FormData();
      formData.append('eye_mode', mode);
      if (mode === 'both') { formData.append('left_file', files.a); formData.append('right_file', files.b); }
      else { formData.append('file', files.a); }

      const btn = document.getElementById('analyze-btn');
      const processing = document.getElementById('processing');
      const results = document.getElementById('results');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner" aria-hidden="true"></span>Analysing…';
      processing.classList.add('active');
      results.style.display = 'none';

      try {
        const res = await fetch('/scan/run', { method: 'POST', headers: { 'Authorization': 'Bearer ' + token }, body: formData });
        let data = {};
        try { data = await res.json(); } catch (e) {}
        if (res.ok) {
          lastScanId = data.id;
          const photo = { left: null, right: null };
          if (mode === 'both') { photo.left = files.a; photo.right = files.b; } else { photo[mode] = files.a; }
          const rows = [];
          if (data.left_diagnosis) rows.push(['LEFT', 'Left eye', data.left_diagnosis, photo.left]);
          if (data.right_diagnosis) rows.push(['RIGHT', 'Right eye', data.right_diagnosis, photo.right]);
          document.getElementById('result-grid').innerHTML = rows.map(([tag, eye, dx, file]) => {
            const d = describe(dx);
            const src = file ? URL.createObjectURL(file) : '';
            return '<div class="result-card ' + d.tone + '">' +
                   '<div class="rc-eye">' + tag + '</div>' +
                   (src ? '<img class="rc-photo" src="' + src + '" alt="' + eye + ' photograph">' : '<span></span>') +
                   '<div><div class="result-headline">' + escapeHtml(d.headline) + '</div>' +
                   (d.detail ? '<div class="result-detail">' + escapeHtml(d.detail) + '</div>' : '') + '</div>' +
                   '<div class="rc-action">' + (d.action ? '<span class="chip ' + d.tone + '">' + escapeHtml(d.action) + '</span>' : '') + '</div>' +
                   '</div>';
          }).join('');
          const now = new Date();
          document.getElementById('result-meta').textContent = 'Alati ' + AI_RELEASE + ' · ' +
            now.toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'}) + ' · report no. ' + data.id;
          results.style.display = 'block';
          results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          loadUsage();
        } else if (res.status === 401) {
          showError('Your session has expired. Please sign in again.');
          setTimeout(logout, 1500);
        } else {
          showError(typeof data.detail === 'string' ? data.detail : 'The scan could not be completed. Please try again with a different photograph.');
        }
      } catch (e) {
        showError('Could not reach the server. Check your connection and try again.');
      } finally {
        processing.classList.remove('active');
        btn.disabled = false;
        btn.textContent = 'Analyze';
      }
    }

    async function downloadReport() {
      if (!lastScanId) return;
      const btn = document.getElementById('report-btn');
      btn.disabled = true;
      try {
        const res = await fetch('/scan/' + lastScanId + '/report', { headers: { 'Authorization': 'Bearer ' + token } });
        if (!res.ok) throw new Error('status ' + res.status);
        const url = URL.createObjectURL(await res.blob());
        const a = document.createElement('a');
        a.href = url; a.download = 'alati_report_' + lastScanId + '.pdf';
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
      } catch (e) {
        showError('The report could not be downloaded. Please try again.');
      } finally {
        btn.disabled = false;
      }
    }

    function resetForm() {
      clearSlot('a'); clearSlot('b');
      hideError();
      document.getElementById('results').style.display = 'none';
      document.getElementById('processing').classList.remove('active');
      lastScanId = null;
    }

    applyMode();
    if (token) loadUsage();
  </script>
</body>
</html>""", current="scan")


# =====================================================================
# Account and history
# =====================================================================
ACCOUNT_HTML = _fill(r"""<!DOCTYPE html>
<html lang="en">
<head>
  {{HEAD}}
  <title>Alati - My account</title>
  <style>
    .ban-notice { display: none; margin-bottom: 18px; }
    .ban-notice.active { display: block; }
    #plan-value { font-family: var(--font); font-weight: 600; font-size: 20px; }
    .premium-soon { margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--line); font-size: 12.5px; color: var(--muted); }
    .premium-soon .head { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-weight: 600; color: var(--ink-2); margin-bottom: 3px; }
    .diagnosis { font-size: 12px; font-weight: 600; padding: 3px 9px; border-radius: 3px; white-space: nowrap; }
    .diagnosis.normal { background: var(--green-bg); color: var(--green); }
    .diagnosis.mild { background: var(--amber-bg); color: var(--amber); }
    .diagnosis.dr { background: var(--red-bg); color: var(--red); }
    .diagnosis.ungradable { background: var(--grey-bg); color: var(--grey); }
    .muted { color: #b4b9c1; }
    .link-btn { background: var(--surface); border: 1px solid var(--line-2); color: var(--blue); border-radius: 4px; padding: 4px 10px;
                font: inherit; font-size: 12.5px; font-weight: 600; cursor: pointer; }
    .link-btn:hover { background: var(--blue-soft); }
    .link-btn:disabled { opacity: 0.5; cursor: wait; }
    td.when { font-family: var(--mono); font-size: 13px; white-space: nowrap; }
  </style>
</head>
<body>
  {{TOPBAR_APP}}

  <main class="page">
    <div class="page-head">
      <div>
        <h1 class="title" id="user-name">My account</h1>
        <p class="sub" id="user-email">-</p>
      </div>
      <span class="chip grey" id="status-pill">Loading…</span>
    </div>

    <div class="notice error ban-notice" id="ban-notice">
      <div class="t">Account suspended</div>
      You can view your previous scans below, but you can't run new scans until the account is reactivated. Contact <a href="mailto:{{CONTACT}}">{{CONTACT}}</a> for assistance.
    </div>

    <div class="panel stats">
      <div class="stat">
        <div class="stat-label">Usage today</div>
        <div class="stat-value" id="usage-value">-</div>
        <div class="stat-sub" id="usage-sub"></div>
        <div class="bar"><div id="usage-bar"></div></div>
      </div>
      <div class="stat">
        <div class="stat-label">Total scans</div>
        <div class="stat-value" id="total-scans">-</div>
        <div class="stat-sub">across all time</div>
      </div>
      <div class="stat">
        <div class="stat-label">Plan</div>
        <div class="stat-value" id="plan-value">-</div>
        <div class="stat-sub" id="plan-sub"></div>
        <div id="premium-soon" class="premium-soon" style="display:none;">
          <div class="head"><span>Premium</span><span class="chip blue">Coming soon</span></div>
          Higher daily scan volumes for clinics and screening programmes. Need more scans now? Email <a href="mailto:{{CONTACT}}">{{CONTACT}}</a>.
        </div>
      </div>
    </div>

    <h2 class="section-title" id="history">Scan history</h2>
    <div class="panel">
      <div class="table-wrap">
        <table class="data" style="min-width: 640px;">
          <thead><tr><th>Date</th><th>Eye</th><th>Left eye</th><th>Right eye</th><th>Report</th></tr></thead>
          <tbody id="my-scans-tbody"><tr><td colspan="5" style="text-align:center; color: var(--muted-2);">Loading…</td></tr></tbody>
        </table>
      </div>
      <div class="pager">
        <span id="my-scans-info">-</span>
        <div class="pager-controls">
          <button type="button" class="btn btn-quiet btn-sm" id="my-prev" onclick="changePage(-1)">← Previous</button>
          <button type="button" class="btn btn-quiet btn-sm" id="my-next" onclick="changePage(1)">Next →</button>
        </div>
      </div>
    </div>
  </main>

  {{FOOTER}}

  <script>
    let myPage = 1, myTotal = 0, myPageSize = 10;
    let planLimits = { free: 5, premium: 50 };
    const token = localStorage.getItem('token');
    if (!token) window.location.replace('/login');
    if (localStorage.getItem('is_admin') === 'true') document.getElementById('admin-btn').style.display = '';

    function logout() { localStorage.removeItem('token'); localStorage.removeItem('is_admin'); window.location.href = '/login'; }
    function escapeHtml(s) {
      if (s == null) return '';
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function describe(dx) {
      const s = dx || '';
      if (!s) return null;
      if (/not gradable/i.test(s)) return { cls: 'ungradable', label: 'Not gradable' };
      if (/not referable/i.test(s)) return { cls: 'mild', label: 'DR, not referable' };
      if (/referable/i.test(s)) return { cls: 'dr', label: 'Referable DR' };
      if (/^normal$/i.test(s.trim())) return { cls: 'normal', label: 'No DR detected' };
      return { cls: 'ungradable', label: s };
    }
    function dxCell(dx, status) {
      if (status === 'failed') return '<span class="diagnosis ungradable">Scan failed</span>';
      const d = describe(dx);
      if (!d) return '<span class="muted">–</span>';
      return '<span class="diagnosis ' + d.cls + '" title="' + escapeHtml(dx) + '">' + escapeHtml(d.label) + '</span>';
    }

    async function loadPlanLimits() {
      try {
        const cfg = await (await fetch('/billing/config')).json();
        if (cfg.free_daily_limit != null) planLimits.free = cfg.free_daily_limit;
        if (cfg.premium_daily_limit != null) planLimits.premium = cfg.premium_daily_limit;
      } catch (e) {}
    }

    async function loadInfo() {
      try {
        const res = await fetch('/user/me', { headers: {'Authorization': 'Bearer ' + token} });
        if (!res.ok) { if (res.status === 401) { logout(); return; } throw new Error('Failed to load account info'); }
        const u = await res.json();
        document.getElementById('user-name').textContent = u.full_name || 'My account';
        document.getElementById('user-email').textContent = u.email;

        const isPremium = (u.tier === 'premium'), isCustom = (u.tier === 'custom');
        document.getElementById('plan-value').textContent = isPremium ? 'Premium' : (isCustom ? 'Custom' : 'Free');
        document.getElementById('plan-sub').textContent = isPremium ? planLimits.premium + ' scans per day'
          : isCustom ? (u.usage_limit === -1 ? 'Unlimited scans' : u.usage_limit + ' scans per day')
          : planLimits.free + ' scans per day';
        document.getElementById('premium-soon').style.display = (isPremium || isCustom) ? 'none' : 'block';

        const pill = document.getElementById('status-pill');
        if (u.is_banned) {
          pill.textContent = 'Suspended'; pill.className = 'chip red';
          document.getElementById('ban-notice').classList.add('active');
          document.getElementById('scan-btn').classList.add('disabled');
        } else {
          pill.textContent = u.is_admin ? 'Active · Administrator' : 'Active'; pill.className = 'chip green';
        }

        const used = u.usage_count || 0, limit = u.usage_limit;
        const unlimited = (limit === -1 || limit === null);
        document.getElementById('usage-value').textContent = unlimited ? String(used) : (used + ' / ' + limit);
        const bar = document.getElementById('usage-bar');
        bar.classList.remove('warn', 'over');
        if (!unlimited) {
          const pct = limit > 0 ? Math.min(100, (used / limit) * 100) : 100;
          bar.style.width = pct + '%';
          if (pct >= 100) { bar.classList.add('over'); document.getElementById('usage-sub').textContent = 'Daily limit reached. Resets at 00:00 UTC.'; }
          else {
            if (pct >= 80) bar.classList.add('warn');
            const left = limit - used;
            document.getElementById('usage-sub').textContent = left + (left === 1 ? ' scan' : ' scans') + ' left today';
          }
        } else {
          bar.style.width = '100%';
          document.getElementById('usage-sub').textContent = 'Unlimited scans';
        }
      } catch (e) { console.error(e); }
    }

    function changePage(delta) {
      const maxPage = Math.max(1, Math.ceil(myTotal / myPageSize));
      myPage = Math.max(1, Math.min(maxPage, myPage + delta));
      loadMyScans();
    }

    async function downloadReport(id, btn) {
      btn.disabled = true;
      try {
        const res = await fetch('/scan/' + id + '/report', { headers: { 'Authorization': 'Bearer ' + token } });
        if (!res.ok) throw new Error('status ' + res.status);
        const url = URL.createObjectURL(await res.blob());
        const a = document.createElement('a');
        a.href = url; a.download = 'alati_report_' + id + '.pdf';
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        btn.textContent = 'PDF';
      } catch (e) { btn.textContent = 'Retry'; }
      finally { btn.disabled = false; }
    }

    async function loadMyScans() {
      const tbody = document.getElementById('my-scans-tbody');
      const info = document.getElementById('my-scans-info');
      try {
        const params = new URLSearchParams({ page: myPage, page_size: myPageSize });
        const res = await fetch('/user/scans?' + params.toString(), { headers: {'Authorization': 'Bearer ' + token} });
        if (!res.ok) { tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color: var(--red);">Could not load your scans (' + res.status + ')</td></tr>'; return; }
        const data = await res.json();
        const scans = data.items || [];
        myTotal = data.total || 0;
        document.getElementById('total-scans').textContent = myTotal;
        const start = myTotal === 0 ? 0 : (myPage - 1) * myPageSize + 1;
        const end = Math.min(myTotal, myPage * myPageSize);
        info.textContent = myTotal === 0 ? 'No scans yet' : `Showing ${start}–${end} of ${myTotal}`;
        const maxPage = Math.max(1, Math.ceil(myTotal / myPageSize));
        document.getElementById('my-prev').disabled = (myPage <= 1);
        document.getElementById('my-next').disabled = (myPage >= maxPage);
        if (scans.length === 0) { tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color: var(--muted-2);">No scans yet. Choose "Scan" above to start.</td></tr>'; return; }
        const eyeName = { left: 'Left', right: 'Right', both: 'Both' };
        tbody.innerHTML = scans.map(s => `
          <tr>
            <td class="when">${new Date(s.created_at).toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'})}</td>
            <td>${escapeHtml(eyeName[s.eye_mode] || s.eye_mode)}</td>
            <td>${dxCell(s.left_diagnosis, s.status)}</td>
            <td>${dxCell(s.right_diagnosis, s.status)}</td>
            <td>${s.status === 'failed' ? '<span class="muted">–</span>' : `<button class="link-btn" onclick="downloadReport(${Number(s.id)}, this)">PDF</button>`}</td>
          </tr>`).join('');
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color: var(--red);">Error: ' + escapeHtml(e.message) + '</td></tr>';
      }
    }

    if (token) { loadPlanLimits().then(loadInfo); loadMyScans(); }
  </script>
</body>
</html>""", current="account")


# =====================================================================
# Admin dashboard
# =====================================================================
DASHBOARD_HTML = _fill(r"""<!DOCTYPE html>
<html lang="en">
<head>
  {{HEAD}}
  <title>Alati - Admin</title>
  <style>
    .stats { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    #model-version { font-size: 24px; }
    .diagnosis { font-weight: 600; }
    .diagnosis.normal { color: var(--green); }
    .diagnosis.dr { color: var(--red); }
    .status { font-size: 12px; padding: 3px 8px; border-radius: 3px; background: var(--green-bg); color: var(--green); }
    .status.banned { background: var(--red-bg); color: var(--red); }
    .user-email { font-weight: 500; }
    td input[type="number"] { width: 76px; padding: 6px 8px; font-size: 13px; }
    td select { width: auto; padding: 6px 8px; font-size: 13px; }
    .scroll { max-height: 520px; overflow: auto; }
    .scroll thead th { position: sticky; top: 0; z-index: 1; }
    .row-actions { display: flex; gap: 6px; }
  </style>
</head>
<body>
  {{TOPBAR_APP}}

  <main class="page">
    <div class="page-head"><div><h1 class="title">Admin</h1><p class="sub">Users, scans and the model in service</p></div></div>

    <div class="panel stats">
      <div class="stat"><div class="stat-label">Total users</div><div class="stat-value" id="user-count">-</div></div>
      <div class="stat"><div class="stat-label">Total scans</div><div class="stat-value" id="scan-count">-</div></div>
      <div class="stat"><div class="stat-label">AI model in service</div><div class="stat-value" id="model-version">-</div><div class="stat-sub" id="model-sub"></div></div>
    </div>

    <h2 class="section-title">Recent scans</h2>
    <div class="panel">
      <div class="toolbar"><h3>Scans</h3><input type="search" id="scans-search" placeholder="Search by email or result" oninput="onScansSearch()" aria-label="Search scans"></div>
      <div class="scroll table-wrap">
        <table class="data" style="min-width: 760px;">
          <thead><tr><th>User</th><th>Eye</th><th>Left eye</th><th>Right eye</th><th>Date</th><th>Status</th></tr></thead>
          <tbody id="scans-tbody"><tr><td colspan="6" style="text-align:center; color: var(--muted-2);">Loading scans…</td></tr></tbody>
        </table>
      </div>
      <div class="pager"><span id="scans-info">-</span><div class="pager-controls">
        <button type="button" class="btn btn-quiet btn-sm" id="scans-prev" onclick="changeScansPage(-1)">← Previous</button>
        <button type="button" class="btn btn-quiet btn-sm" id="scans-next" onclick="changeScansPage(1)">Next →</button>
      </div></div>
    </div>

    <h2 class="section-title">Users</h2>
    <div class="panel">
      <div class="toolbar">
        <h3>All users</h3>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap: wrap;">
          <input type="search" id="users-search" placeholder="Search by email" oninput="onUsersSearch()" aria-label="Search users">
          <button type="button" class="btn btn-warn btn-sm" onclick="resetLegacyAccounts()" title="Resets stale Free-tier limits and manually-granted Premium accounts with no real subscription back to the current Free default. Custom accounts are left alone.">Reset legacy accounts</button>
        </div>
      </div>
      <div class="scroll table-wrap">
        <table class="data" style="min-width: 760px;">
          <thead><tr><th>Email</th><th>Usage</th><th>Limit</th><th>Plan</th><th>Status</th><th>Action</th></tr></thead>
          <tbody id="users-tbody"><tr><td colspan="6" style="text-align:center; color: var(--muted-2);">Loading users…</td></tr></tbody>
        </table>
      </div>
      <div class="pager"><span id="users-info">-</span><div class="pager-controls">
        <button type="button" class="btn btn-quiet btn-sm" id="users-prev" onclick="changeUsersPage(-1)">← Previous</button>
        <button type="button" class="btn btn-quiet btn-sm" id="users-next" onclick="changeUsersPage(1)">Next →</button>
      </div></div>
    </div>
  </main>

  {{FOOTER}}

  <script>
    if (!localStorage.getItem('token')) window.location.replace('/login');
    else if (localStorage.getItem('is_admin') !== 'true') window.location.replace('/scan');
    document.getElementById('admin-btn').style.display = '';

    function logout() { localStorage.removeItem('token'); localStorage.removeItem('is_admin'); window.location.href = '/login'; }
    function escapeHtml(s) {
      if (s == null) return '';
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    let scansPage = 1, scansQuery = '', scansTotal = 0, scansPageSize = 10;
    let usersPage = 1, usersQuery = '', usersTotal = 0, usersPageSize = 10;
    let scansSearchTimer = null, usersSearchTimer = null;
    let planDefaults = { free: 5, premium: 50 };

    async function loadPlanDefaults() {
      try {
        const cfg = await (await fetch('/billing/config')).json();
        if (cfg.free_daily_limit != null) planDefaults.free = cfg.free_daily_limit;
        if (cfg.premium_daily_limit != null) planDefaults.premium = cfg.premium_daily_limit;
      } catch (e) {}
    }
    async function loadModel() {
      try {
        const m = await (await fetch('/model_info')).json();
        document.getElementById('model-version').textContent = 'Alati ' + (m.release || '{{AI_RELEASE}}');
        const r = (m.measured || {}).referable;
        if (r) document.getElementById('model-sub').textContent =
          'Referable DR: sensitivity ' + r.sensitivity + '%, specificity ' + r.specificity + '% (APTOS, external)';
      } catch (e) {}
    }

    function onTierChange(id) {
      const tier = document.getElementById('tier-' + id).value;
      if (tier === 'custom') return;
      document.getElementById('limit-' + id).value = (tier === 'premium') ? planDefaults.premium : planDefaults.free;
    }

    async function resetLegacyAccounts() {
      if (!confirm("Reset stale Free-tier usage limits and manually-granted Premium accounts (no real subscription) back to the current Free plan default?\n\nAccounts set to Custom are left untouched.")) return;
      const token = localStorage.getItem('token');
      try {
        const res = await fetch('/admin/users/reset-legacy-free', { method: 'POST', headers: {'Authorization': 'Bearer ' + token} });
        if (res.ok) { const data = await res.json(); alert(`Reset ${data.reset_count} account(s) to Free.`); loadUsers(); }
        else alert('Failed to reset accounts: ' + res.status);
      } catch (e) { alert('Error: ' + e.message); }
    }

    function onScansSearch() { clearTimeout(scansSearchTimer); scansSearchTimer = setTimeout(() => { scansQuery = document.getElementById('scans-search').value.trim(); scansPage = 1; loadScans(); }, 300); }
    function onUsersSearch() { clearTimeout(usersSearchTimer); usersSearchTimer = setTimeout(() => { usersQuery = document.getElementById('users-search').value.trim(); usersPage = 1; loadUsers(); }, 300); }
    function changeScansPage(d) { const m = Math.max(1, Math.ceil(scansTotal / scansPageSize)); scansPage = Math.max(1, Math.min(m, scansPage + d)); loadScans(); }
    function changeUsersPage(d) { const m = Math.max(1, Math.ceil(usersTotal / usersPageSize)); usersPage = Math.max(1, Math.min(m, usersPage + d)); loadUsers(); }

    function dxClass(dx) { return dx && dx.toLowerCase() === 'normal' ? 'normal' : 'dr'; }

    async function loadScans() {
      const token = localStorage.getItem('token');
      const tbody = document.getElementById('scans-tbody'), info = document.getElementById('scans-info');
      try {
        const params = new URLSearchParams({ q: scansQuery, page: scansPage, page_size: scansPageSize });
        const res = await fetch('/admin/scans?' + params.toString(), { headers: {'Authorization': 'Bearer ' + token} });
        if (!res.ok) { tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--red);">Failed to load scans (' + res.status + ')</td></tr>'; document.getElementById('scan-count').textContent = '!'; info.textContent = ''; return; }
        const data = await res.json(); const scans = data.items || []; scansTotal = data.total || 0;
        document.getElementById('scan-count').textContent = scansTotal;
        const start = scansTotal === 0 ? 0 : (scansPage - 1) * scansPageSize + 1, end = Math.min(scansTotal, scansPage * scansPageSize);
        info.textContent = scansTotal === 0 ? 'No results' : `Showing ${start}–${end} of ${scansTotal}`;
        const maxPage = Math.max(1, Math.ceil(scansTotal / scansPageSize));
        document.getElementById('scans-prev').disabled = (scansPage <= 1);
        document.getElementById('scans-next').disabled = (scansPage >= maxPage);
        if (scans.length === 0) { tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--muted-2);">' + (scansQuery ? 'No scans match your search' : 'No scans yet') + '</td></tr>'; return; }
        tbody.innerHTML = scans.map(s => `
          <tr>
            <td class="user-email">${escapeHtml(s.user_email)}</td>
            <td>${escapeHtml(s.eye_mode)}</td>
            <td><span class="diagnosis ${dxClass(s.left_diagnosis)}">${escapeHtml(s.left_diagnosis || '-')}</span></td>
            <td><span class="diagnosis ${dxClass(s.right_diagnosis)}">${escapeHtml(s.right_diagnosis || '-')}</span></td>
            <td class="mono" style="font-size:13px; white-space:nowrap;">${new Date(s.created_at).toLocaleString()}</td>
            <td><span class="status ${s.status === 'failed' ? 'banned' : ''}">${escapeHtml(s.status)}</span></td>
          </tr>`).join('');
      } catch (e) { tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--red);">Error: ' + escapeHtml(e.message) + '</td></tr>'; }
    }

    async function loadUsers() {
      const token = localStorage.getItem('token');
      const tbody = document.getElementById('users-tbody'), info = document.getElementById('users-info');
      try {
        const params = new URLSearchParams({ q: usersQuery, page: usersPage, page_size: usersPageSize });
        const res = await fetch('/admin/users?' + params.toString(), { headers: {'Authorization': 'Bearer ' + token} });
        if (!res.ok) { tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--red);">Failed to load users (' + res.status + ')</td></tr>'; info.textContent = ''; return; }
        const data = await res.json(); const users = data.items || []; usersTotal = data.total || 0;
        document.getElementById('user-count').textContent = usersTotal;
        const start = usersTotal === 0 ? 0 : (usersPage - 1) * usersPageSize + 1, end = Math.min(usersTotal, usersPage * usersPageSize);
        info.textContent = usersTotal === 0 ? 'No results' : `Showing ${start}–${end} of ${usersTotal}`;
        const maxPage = Math.max(1, Math.ceil(usersTotal / usersPageSize));
        document.getElementById('users-prev').disabled = (usersPage <= 1);
        document.getElementById('users-next').disabled = (usersPage >= maxPage);
        if (users.length === 0) { tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--muted-2);">' + (usersQuery ? 'No users match your search' : 'No users yet') + '</td></tr>'; return; }
        tbody.innerHTML = users.map(u => {
          const id = Number(u.id);
          const used = u.usage_count || 0, limit = u.usage_limit;
          const limitDisplay = (limit === -1 || limit === null) ? '∞' : limit;
          const over = (limit >= 0 && used >= limit), near = (limit > 0 && used >= limit * 0.8);
          const color = over ? 'var(--red)' : (near ? 'var(--amber)' : 'var(--green)');
          const tier = (u.tier || 'free').toLowerCase();
          return `
          <tr id="user-row-${id}">
            <td class="user-email">${escapeHtml(u.email)}</td>
            <td class="mono" style="font-weight:500; color:${color};">${used} / ${limitDisplay}</td>
            <td><input type="number" id="limit-${id}" value="${Number(limit)}" title="Use -1 for unlimited" aria-label="Daily limit"></td>
            <td><select id="tier-${id}" onchange="onTierChange(${id})" aria-label="Plan">
                <option value="free" ${tier === 'free' ? 'selected' : ''}>Free</option>
                <option value="premium" ${tier === 'premium' ? 'selected' : ''}>Premium</option>
                <option value="custom" ${tier === 'custom' ? 'selected' : ''}>Custom</option>
            </select></td>
            <td><select id="banned-${id}" aria-label="Status">
                <option value="0" ${!u.is_banned ? 'selected' : ''}>Active</option>
                <option value="1" ${u.is_banned ? 'selected' : ''}>Suspended</option>
            </select></td>
            <td><div class="row-actions">
              <button type="button" class="btn btn-sm" onclick="updateUser(${id})">Save</button>
              <button type="button" class="btn btn-warn btn-sm" onclick="resetUsage(${id})" title="Reset today's usage to 0">Reset usage</button>
            </div></td>
          </tr>`;
        }).join('');
      } catch (e) { tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--red);">Error: ' + escapeHtml(e.message) + '</td></tr>'; }
    }

    async function updateUser(id) {
      const token = localStorage.getItem('token');
      const params = new URLSearchParams({ usage_limit: parseInt(document.getElementById('limit-' + id).value),
                                           is_banned: parseInt(document.getElementById('banned-' + id).value),
                                           tier: document.getElementById('tier-' + id).value });
      try {
        const res = await fetch(`/admin/users/${id}?` + params.toString(), { method: 'PATCH', headers: {'Authorization': 'Bearer ' + token} });
        if (res.ok) {
          const row = document.getElementById('user-row-' + id);
          if (row) { row.style.transition = 'background 0.3s'; row.style.background = 'var(--green-bg)'; setTimeout(() => { row.style.background = ''; }, 800); }
          loadUsers();
        } else alert('Failed to update user: ' + res.status);
      } catch (e) { alert('Error: ' + e.message); }
    }

    async function resetUsage(id) {
      if (!confirm("Reset this user's usage count to 0?")) return;
      const token = localStorage.getItem('token');
      try {
        const res = await fetch(`/admin/users/${id}/reset-usage`, { method: 'POST', headers: {'Authorization': 'Bearer ' + token} });
        if (res.ok) loadUsers(); else alert('Failed to reset usage');
      } catch (e) { alert('Error: ' + e.message); }
    }

    loadPlanDefaults(); loadModel(); loadScans(); loadUsers();
  </script>
</body>
</html>""", current="admin")


# =====================================================================
# Document pages (legal, pricing), password reset, email verification
# =====================================================================
def document_page(title: str, body_html: str, last_updated: str) -> str:
    return _fill("""<!DOCTYPE html>
<html lang="en">
<head>
  {{HEAD}}
  <title>Alati - """ + title + """</title>
</head>
<body>
  {{TOPBAR_PUBLIC}}
  <main class="page narrow">
    <div class="page-head"><div><h1 class="title">""" + title + """</h1>
      <p class="meta" style="margin-top:6px;">Alati · diabetic retinopathy screening · last updated """ + last_updated + """</p></div></div>
    <article class="panel doc">
""" + body_html + """
    </article>
    <p class="fine" style="margin-top:14px;"><a href="/login">← Back to Alati</a></p>
  </main>
  {{FOOTER}}
</body>
</html>""")


RESET_PASSWORD_HTML = _fill(r"""<!DOCTYPE html>
<html lang="en">
<head>
  {{HEAD}}
  <title>Alati - Choose a new password</title>
  <style>
    .msg { display: none; margin-bottom: 16px; }
    .msg.error, .msg.ok { display: block; }
  </style>
</head>
<body>
  {{TOPBAR_PUBLIC}}
  <main class="page narrow" style="max-width: 460px;">
    <div class="page-head"><div><h1 class="title">Choose a new password</h1><p class="sub">The link from your email works once and expires after 1 hour.</p></div></div>
    <form id="f" class="panel" style="padding: 24px;" novalidate>
      <div class="msg notice" id="msg"></div>
      <div class="field"><label class="label" for="pw">New password</label><input type="password" id="pw" autocomplete="new-password" placeholder="At least 8 characters"></div>
      <div class="field"><label class="label" for="pw2">Repeat new password</label><input type="password" id="pw2" autocomplete="new-password"></div>
      <button type="submit" class="btn btn-block" id="btn">Save new password</button>
      <p class="fine" style="text-align:center; margin-top: 14px;"><a href="/login">Back to sign in</a></p>
    </form>
  </main>
  {{FOOTER}}
  <script>
    const token = new URLSearchParams(location.search).get('token') || '';
    const msg = document.getElementById('msg');
    function say(cls, text) { msg.className = 'msg notice ' + cls; msg.textContent = text; }
    if (!token) say('error', 'This link is incomplete. Request a new reset link from the sign-in page.');
    document.getElementById('f').addEventListener('submit', async (e) => {
      e.preventDefault();
      const pw = document.getElementById('pw').value, pw2 = document.getElementById('pw2').value;
      if (pw.length < 8) return say('error', 'Password must be at least 8 characters.');
      if (pw !== pw2) return say('error', 'The two passwords do not match.');
      const btn = document.getElementById('btn'); btn.disabled = true;
      try {
        const res = await fetch('/auth/reset-password', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ token, password: pw }) });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          say('ok', data.message || 'Your password has been changed.');
          document.getElementById('pw').value = ''; document.getElementById('pw2').value = '';
          btn.textContent = 'Go to sign in'; btn.disabled = false;
          btn.onclick = (ev) => { ev.preventDefault(); location.href = '/login'; };
          return;
        }
        say('error', typeof data.detail === 'string' ? data.detail : 'Could not change the password. Please try again.');
      } catch (err) { say('error', 'Could not reach the server. Check your connection and try again.'); }
      btn.disabled = false;
    });
  </script>
</body>
</html>""")


def verification_page(success: bool, message: str) -> str:
    import html as _html
    tone = "ok" if success else "error"
    heading = "Email verified" if success else "Verification failed"
    return _fill("""<!DOCTYPE html>
<html lang="en">
<head>
  {{HEAD}}
  <title>Alati - Email verification</title>
</head>
<body>
  {{TOPBAR_PUBLIC}}
  <main class="page narrow" style="max-width: 520px;">
    <div class="page-head"><div><h1 class="title">""" + heading + """</h1></div></div>
    <div class="notice """ + tone + """" style="margin-bottom: 18px;">""" + _html.escape(message) + """</div>
    <a class="btn" href="/login">Go to sign in</a>
  </main>
  {{FOOTER}}
</body>
</html>""")


def email_html(title: str, intro: str, button_text: str, link: str, after: str) -> str:
    """Plain, flat email in design A: wordmark, rule, text, one button, the raw link."""
    import html as _html
    t, i, b, a, l = (_html.escape(x) for x in (title, intro, button_text, after, link))
    return f"""<html><body style="margin:0; padding:24px; background:#f5f6f8; font-family:'Public Sans', Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:#17202b;">
  <div style="max-width:520px; margin:0 auto; background:#ffffff; border:1px solid #dfe4ea; border-radius:6px;">
    <div style="padding:16px 24px; border-bottom:1px solid #dfe4ea; font-weight:700; font-size:17px;">
      <span style="display:inline-block; width:12px; height:12px; border:3px solid #1f4f8a; border-radius:50%; vertical-align:-2px; margin-right:6px;"></span>Alati
    </div>
    <div style="padding:24px;">
      <h1 style="margin:0 0 12px; font-size:20px;">{t}</h1>
      <p style="margin:0 0 20px; font-size:15px; line-height:1.6; color:#2b3540;">{i}</p>
      <p style="margin:0 0 22px;"><a href="{l}" style="background:#1f4f8a; color:#ffffff; padding:11px 22px; border-radius:4px; text-decoration:none; font-weight:600; display:inline-block;">{b}</a></p>
      <p style="margin:0 0 4px; font-size:13px; color:#6b7684;">Or copy this link into your browser:</p>
      <p style="margin:0 0 20px; font-size:13px; color:#1f4f8a; word-break:break-all;">{l}</p>
      <p style="margin:0; padding-top:14px; border-top:1px solid #edf0f3; font-size:12.5px; color:#6b7684;">{a}</p>
    </div>
  </div>
</body></html>"""
