#!/bin/bash

echo "=== Final Fix Test - ElectronAPI Declaration Error ==="
echo "Testing the fix for 'electronAPI' has already been declared error"
echo ""

source env/bin/activate
# Start backend server
echo "Starting backend server..."
cd serve
python api-local.py --host localhost --port 8001 &
SERVER_PID=$!
cd ..

# Wait for server to start
echo "Waiting for server to start..."
sleep 3

# Test server is running
echo "Testing server connection..."
curl -s http://localhost:8001 > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ Server is running"
else
    echo "✗ Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo ""
echo "Starting Electron app..."
echo "Expected behavior:"
echo "1. Main window should load without JavaScript errors"
echo "2. Translation window should appear without 'electronAPI already declared' error"
echo "3. Translation window should be functional with status updates"
echo "4. Developer tools should show NO JavaScript errors"
echo ""

# Start Electron
ELECTRON_IS_DEV=true ./node_modules/.bin/electron electron/main-simple.js &
ELECTRON_PID=$!

# Wait a bit for Electron to start
sleep 5

echo ""
echo "Electron app should now be running..."
echo "Please check:"
echo "1. No JavaScript errors in browser console"
echo "2. Translation window appears and shows status properly"  
echo "3. You can load a model and the status updates correctly"
echo ""
echo "Press Ctrl+C to stop both server and Electron app"

# Wait for user to stop
wait $ELECTRON_PID

# Clean up
echo ""
echo "Cleaning up..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "Test completed!" 