// Toggle between login and register forms
document.getElementById('showRegister').addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('loginForm').style.display = 'none';
    document.getElementById('registerForm').style.display = 'block';
    clearMessages();
});

document.getElementById('showLogin').addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('registerForm').style.display = 'none';
    document.getElementById('loginForm').style.display = 'block';
    clearMessages();
});

// Register form submission
document.getElementById('registerFormElement').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    const passwordConfirm = document.getElementById('registerPasswordConfirm').value;
    
    const errorEl = document.getElementById('registerError');
    const successEl = document.getElementById('registerSuccess');
    
    // Clear previous messages
    errorEl.classList.remove('show');
    successEl.classList.remove('show');
    
    // Validate passwords match
    if (password !== passwordConfirm) {
        showError(errorEl, 'Passwords do not match');
        return;
    }
    
    // Disable button and show loader
    const btn = document.getElementById('registerBtn');
    const btnText = document.getElementById('registerBtnText');
    const btnLoader = document.getElementById('registerLoader');
    
    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    
    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            successEl.textContent = 'Account created! Redirecting to login...';
            successEl.classList.add('show');
            
            setTimeout(() => {
                document.getElementById('showLogin').click();
            }, 2000);
        } else {
            const errorMsg = Array.isArray(data.error) 
                ? data.error.map(e => e.msg).join(', ')
                : data.error;
            showError(errorEl, errorMsg);
        }
    } catch (error) {
        showError(errorEl, 'Network error. Please try again.');
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

// Login form submission
document.getElementById('loginFormElement').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    
    const errorEl = document.getElementById('loginError');
    errorEl.classList.remove('show');
    
    // Disable button and show loader
    const btn = document.getElementById('loginBtn');
    const btnText = document.getElementById('loginBtnText');
    const btnLoader = document.getElementById('loginLoader');
    
    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Store token
            localStorage.setItem('access_token', data.access_token);
            localStorage.setItem('user_email', data.email);
            
            setTimeout(() => {
                window.location.href = '/chat';
            }, 100);
            // Redirect to chat
            window.location.href = '/chat';
        } else {
            showError(errorEl, data.error || 'Invalid credentials');
        }
    } catch (error) {
        showError(errorEl, 'Network error. Please try again.');
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

function showError(element, message) {
    element.textContent = message;
    element.classList.add('show');
}

function clearMessages() {
    document.querySelectorAll('.error-message, .success-message').forEach(el => {
        el.classList.remove('show');
    });
}