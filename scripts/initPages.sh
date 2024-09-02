#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Create docs directory if it doesn't exist
mkdir -p docs

# Function to create an HTML file with content
create_page() {
    local file="$1"
    local title="$2"
    local content="$3"
    cat > "docs/$file" <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$title - robinBots</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { padding-top: 60px; }
        .jumbotron { padding: 2rem 1rem; background-color: #e9ecef; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="index.html">robinBots</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item"><a class="nav-link" href="index.html">Home</a></li>
                    <li class="nav-item"><a class="nav-link" href="features.html">Features</a></li>
                    <li class="nav-item"><a class="nav-link" href="pricing.html">Pricing</a></li>
                    <li class="nav-item"><a class="nav-link" href="contact.html">Contact</a></li>
                </ul>
            </div>
        </div>
    </nav>

    <main class="container">
        $content
    </main>

    <footer class="footer mt-auto py-3 bg-light">
        <div class="container text-center">
            <span class="text-muted">© 2023 robinBots. All rights reserved.</span>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
EOF
    print_color $BLUE "Created $file"
}

# Create index.html
print_color $GREEN "Creating index.html..."
create_page "index.html" "Home" '
<div class="jumbotron">
    <h1 class="display-4">Welcome to robinBots</h1>
    <p class="lead">Automate your trading with advanced AI and machine learning algorithms.</p>
    <hr class="my-4">
    <p>robinBots connects to your Robinhood account and executes trades based on sophisticated analysis and risk management strategies.</p>
    <a class="btn btn-primary btn-lg" href="features.html" role="button">Learn More</a>
</div>

<div class="row mt-5">
    <div class="col-md-4">
        <h2>Smart Trading</h2>
        <p>Our AI algorithms analyze market trends and execute trades at optimal times.</p>
    </div>
    <div class="col-md-4">
        <h2>Risk Management</h2>
        <p>Advanced risk management techniques protect your investments.</p>
    </div>
    <div class="col-md-4">
        <h2>Easy Setup</h2>
        <p>Connect your Robinhood account and start automated trading in minutes.</p>
    </div>
</div>
'

# Create features.html
print_color $GREEN "Creating features.html..."
create_page "features.html" "Features" '
<h1 class="mb-4">robinBots Features</h1>

<div class="row">
    <div class="col-md-6 mb-4">
        <h3>AI-Powered Analysis</h3>
        <p>Our machine learning algorithms analyze vast amounts of market data to identify profitable trading opportunities.</p>
    </div>
    <div class="col-md-6 mb-4">
        <h3>Real-Time Trading</h3>
        <p>Execute trades automatically in real-time based on market conditions and our AI predictions.</p>
    </div>
    <div class="col-md-6 mb-4">
        <h3>Risk Management</h3>
        <p>Implement stop-loss orders, position sizing, and portfolio diversification to manage risk effectively.</p>
    </div>
    <div class="col-md-6 mb-4">
        <h3>Performance Tracking</h3>
        <p>Monitor your trading performance with detailed analytics and reports.</p>
    </div>
    <div class="col-md-6 mb-4">
        <h3>Customizable Strategies</h3>
        <p>Tailor trading strategies to your risk tolerance and investment goals.</p>
    </div>
    <div class="col-md-6 mb-4">
        <h3>Secure Integration</h3>
        <p>Connect securely to your Robinhood account with bank-level encryption.</p>
    </div>
</div>
'

# Create pricing.html
print_color $GREEN "Creating pricing.html..."
create_page "pricing.html" "Pricing" '
<h1 class="mb-4">Pricing Plans</h1>

<div class="row row-cols-1 row-cols-md-3 mb-3 text-center">
    <div class="col">
        <div class="card mb-4 rounded-3 shadow-sm">
            <div class="card-header py-3">
                <h4 class="my-0 fw-normal">Basic</h4>
            </div>
            <div class="card-body">
                <h1 class="card-title pricing-card-title">$29<small class="text-muted fw-light">/mo</small></h1>
                <ul class="list-unstyled mt-3 mb-4">
                    <li>Up to $10,000 portfolio size</li>
                    <li>Basic AI trading strategies</li>
                    <li>Email support</li>
                </ul>
                <button type="button" class="w-100 btn btn-lg btn-outline-primary">Sign up</button>
            </div>
        </div>
    </div>
    <div class="col">
        <div class="card mb-4 rounded-3 shadow-sm">
            <div class="card-header py-3">
                <h4 class="my-0 fw-normal">Pro</h4>
            </div>
            <div class="card-body">
                <h1 class="card-title pricing-card-title">$79<small class="text-muted fw-light">/mo</small></h1>
                <ul class="list-unstyled mt-3 mb-4">
                    <li>Up to $50,000 portfolio size</li>
                    <li>Advanced AI trading strategies</li>
                    <li>Priority email support</li>
                </ul>
                <button type="button" class="w-100 btn btn-lg btn-primary">Get started</button>
            </div>
        </div>
    </div>
    <div class="col">
        <div class="card mb-4 rounded-3 shadow-sm border-primary">
            <div class="card-header py-3 text-bg-primary border-primary">
                <h4 class="my-0 fw-normal">Enterprise</h4>
            </div>
            <div class="card-body">
                <h1 class="card-title pricing-card-title">$199<small class="text-muted fw-light">/mo</small></h1>
                <ul class="list-unstyled mt-3 mb-4">
                    <li>Unlimited portfolio size</li>
                    <li>Custom AI trading strategies</li>
                    <li>24/7 phone & email support</li>
                </ul>
                <button type="button" class="w-100 btn btn-lg btn-primary">Contact us</button>
            </div>
        </div>
    </div>
</div>
'

# Create contact.html
print_color $GREEN "Creating contact.html..."
create_page "contact.html" "Contact" '
<h1 class="mb-4">Contact Us</h1>

<div class="row">
    <div class="col-md-6">
        <form>
            <div class="mb-3">
                <label for="name" class="form-label">Name</label>
                <input type="text" class="form-control" id="name" required>
            </div>
            <div class="mb-3">
                <label for="email" class="form-label">Email address</label>
                <input type="email" class="form-control" id="email" required>
            </div>
            <div class="mb-3">
                <label for="message" class="form-label">Message</label>
                <textarea class="form-control" id="message" rows="5" required></textarea>
            </div>
            <button type="submit" class="btn btn-primary">Send Message</button>
        </form>
    </div>
    <div class="col-md-6">
        <h3>Get in Touch</h3>
        <p>Have questions about robinBots? We're here to help! Fill out the form or contact us using the information below.</p>
        <ul class="list-unstyled">
            <li>Email: support@robinbots.com</li>
            <li>Phone: (555) 123-4567</li>
            <li>Address: 123 AI Street, Tech City, TC 12345</li>
        </ul>
    </div>
</div>
'

print_color $GREEN "GitHub Pages content creation complete!"
print_color $BLUE "The following files have been created in the docs/ directory:"
ls -1 docs/

print_color $GREEN "Next steps:"
echo "1. Review and customize the generated HTML files as needed."
echo "2. Add any additional pages or content specific to your robinBots offering."
echo "3. Set up GitHub Pages in your repository settings to use the 'docs' folder."
echo "4. Consider adding a custom domain for a more professional look."

print_color $BLUE "Remember to commit these changes to your repository:"
echo "git add docs/"
echo "git commit -m \"Add GitHub Pages content for robinBots subscription service\""
echo "git push origin main"

print_color $GREEN "GitHub Pages content creation process completed successfully!"