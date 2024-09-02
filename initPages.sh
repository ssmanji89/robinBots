#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Create necessary directories
mkdir -p docs/js
mkdir -p docs/css
mkdir -p server

# Create server-side files
cat > server/stripe_server.js <<EOF
const express = require('express');
const stripe = require('stripe')('sk_test_51PuSMQP7HsCzcZI7FrQ5Zmy2PRh9ovxtq9aqSftS9oG8sHKTGddy3GwuhyEcEJZP68YsLV9WY2aCKhmBL5qjgJjI00KpH8WQxH');
const app = express();

app.use(express.static('docs'));
app.use(express.json());

const DOMAIN = 'http://localhost:4242';

app.post('/create-checkout-session', async (req, res) => {
  const prices = {
    'basic': 'price_1234', // Replace with your actual price IDs
    'pro': 'price_5678',
    'enterprise': 'price_9012'
  };

  const session = await stripe.checkout.sessions.create({
    mode: 'subscription',
    line_items: [
      {
        price: prices[req.body.plan],
        quantity: 1,
      },
    ],
    success_url: \`\${DOMAIN}/success.html?session_id={CHECKOUT_SESSION_ID}\`,
    cancel_url: \`\${DOMAIN}/cancel.html\`,
  });

  res.json({ clientSecret: session.client_secret });
});

app.get('/session-status', async (req, res) => {
  const session = await stripe.checkout.sessions.retrieve(req.query.session_id);
  const customer = await stripe.customers.retrieve(session.customer);

  res.json({
    status: session.status,
    payment_status: session.payment_status,
    customer_email: customer.email
  });
});

app.listen(4242, () => console.log('Running on port 4242'));
EOF

print_color $BLUE "Created server/stripe_server.js"

# Create client-side JavaScript
cat > docs/js/stripe_checkout.js <<EOF
const stripe = Stripe('pk_test_51PuSMQP7HsCzcZI7mcWSEaJWqPoiMQQyG6UkveFYHE1AcY8o55SKswPIMT8a9zmCw11cYlbdAjJjzwt0xkjA66Fh00qcLmC5Hg');

let elements;

initialize();

async function initialize() {
  const fetchClientSecret = async () => {
    const response = await fetch("/create-checkout-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan: document.getElementById('plan').value })
    });
    const { clientSecret } = await response.json();
    return clientSecret;
  };

  const checkout = await stripe.initEmbeddedCheckout({
    fetchClientSecret,
  });

  checkout.mount('#checkout');
}

document.querySelector("#submit").addEventListener("click", initialize);
EOF

print_color $BLUE "Created docs/js/stripe_checkout.js"

cat > docs/js/stripe_return.js <<EOF
initialize();

async function initialize() {
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const sessionId = urlParams.get('session_id');
  const response = await fetch(\`/session-status?session_id=\${sessionId}\`);
  const session = await response.json();

  if (session.status == 'open') {
    window.location.replace('/');
  } else if (session.status == 'complete') {
    document.getElementById('success').classList.remove('hidden');
    document.getElementById('customer-email').textContent = session.customer_email;
  }
}
EOF

print_color $BLUE "Created docs/js/stripe_return.js"

# Create HTML snippets
cat > docs/stripe_subscription_snippet.html <<EOF
<div id="subscribe" class="mt-5">
    <h2>Subscribe Now</h2>
    <p>Ready to start your journey with robinBots? Choose your plan and get started today!</p>
    <form id="payment-form">
        <div class="mb-3">
            <label for="plan" class="form-label">Select your plan:</label>
            <select class="form-select" id="plan" name="plan" required>
                <option value="">Choose a plan...</option>
                <option value="basic">Basic ($29/mo)</option>
                <option value="pro">Pro ($79/mo)</option>
                <option value="enterprise">Enterprise ($199/mo)</option>
            </select>
        </div>
        <div id="checkout"></div>
        <button id="submit" type="button" class="btn btn-primary">Subscribe</button>
    </form>
</div>
EOF

print_color $BLUE "Created docs/stripe_subscription_snippet.html"

cat > docs/stripe_success.html <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Subscription Success - robinBots</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="css/style.css">
    <script src="js/stripe_return.js" defer></script>
</head>
<body>
    <main class="container">
        <div id="success" class="hidden">
            <h1>Subscription Successful!</h1>
            <p>Thank you for subscribing to robinBots. An email confirmation has been sent to <span id="customer-email"></span>.</p>
            <a href="/" class="btn btn-primary">Return to Home</a>
        </div>
    </main>
</body>
</html>
EOF

print_color $BLUE "Created docs/stripe_success.html"

cat > docs/stripe_cancel.html <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Subscription Cancelled - robinBots</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <main class="container">
        <h1>Subscription Cancelled</h1>
        <p>Your subscription process was cancelled. If this was a mistake, please try again.</p>
        <a href="/" class="btn btn-primary">Return to Home</a>
    </main>
</body>
</html>
EOF

print_color $BLUE "Created docs/stripe_cancel.html"

print_color $GREEN "Stripe integration files created!"
print_color $GREEN "Next steps for manual integration:"

echo "1. In your existing index.html file, add the following in the <head> section:"
echo "   <script src=\"https://js.stripe.com/v3/\"></script>"
echo "   <script src=\"js/stripe_checkout.js\" defer></script>"

echo "2. In your existing index.html file, replace the subscription form with the content from docs/stripe_subscription_snippet.html"

echo "3. Add stripe_success.html and stripe_cancel.html to your docs directory"

echo "4. In your server code, integrate the content from stripe_server.js:"
echo "   - Install required dependencies: npm install express stripe"
echo "   - Add the new routes for creating checkout sessions and checking session status"

echo "5. Update the Stripe public key in docs/js/stripe_checkout.js and the secret key in server/stripe_server.js"

echo "6. Update the price IDs in server/stripe_server.js with your actual Stripe price IDs"

echo "7. Implement proper error handling and logging in the server code"

echo "8. Set up Stripe webhooks to handle subscription lifecycle events"

echo "9. Implement a customer portal for managing subscriptions"

echo "10. Ensure all sensitive data is properly secured and that your server meets PCI-DSS requirements"

print_color $BLUE "To test the Stripe integration locally:"
echo "1. Start your server with the new Stripe routes"
echo "2. Open your index.html file in a browser"
echo "3. Test the subscription process using Stripe test cards"

print_color $GREEN "Remember to thoroughly test the integration and consult with security experts to ensure full PCI-DSS compliance before going live."