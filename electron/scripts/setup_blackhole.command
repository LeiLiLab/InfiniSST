#!/bin/bash

echo "🚀 BlackHole + Multi-Output Setup (Step 1: Install & Reboot)"

# 确保路径正确（兼容 Intel 与 Apple Silicon）
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# 安装 Homebrew（如果没装）
if ! command -v brew &> /dev/null; then
  echo "🍺 Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  echo "✅ Homebrew already installed."
fi

need_reboot=false

# 检查 BlackHole 是否新安装
if ! brew list --cask | grep -q "blackhole-16ch"; then
  echo "🎧 Installing BlackHole 16ch..."
  brew install --cask blackhole-16ch
  echo "📢 System restart is required to activate BlackHole."
  need_reboot=true
else
  echo "✅ BlackHole 16ch already installed."
fi

# 检查 switchaudio-osx 是否安装
if ! command -v SwitchAudioSource &> /dev/null; then
  echo "🎚 Installing switchaudio-osx..."
  brew install switchaudio-osx
else
  echo "✅ switchaudio-osx already installed."
fi

# 判断是否需要重启
if [[ "$need_reboot" == true ]]; then
  echo ""
  echo "⚠️ BlackHole has been installed successfully!"
  echo "📢 System restart is required to activate BlackHole driver."
  echo ""
  echo "🔄 Your Mac will restart automatically in 3 seconds..."
  echo "💡 Press Ctrl+C to cancel automatic restart"
  echo ""
  
  # 倒数3秒
  for i in {3..1}; do
    echo "⏰ Restarting in $i seconds..."
    sleep 1
  done
  
  echo ""
  echo "🚀 Restarting Mac now..."
  echo "✨ After restart, click 'Capture System Audio' again to continue setup."
  echo ""
  
  # 执行重启
  sudo shutdown -r now
else
  echo "✅ All components already installed. No restart required."

  echo ""
  echo "🔄 BlackHole + Multi-Output Setup (Step 2: Configure Output)"

  # 确保路径正确
  export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

  echo "📂 Opening 'Audio MIDI Setup'..."
  open -a "Audio MIDI Setup"

  echo ""
  echo -e "\033[1;33m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
  echo -e "\033[1;31m📣 IMPORTANT: Please do the following in the Audio MIDI Setup window:\033[0m"
  echo -e "\033[1;33m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
  echo ""
  echo -e "\033[1;36m  1. Click ➕ in the lower-left corner\033[0m"
  echo -e "\033[1;36m  2. Choose 'Create Multi-Output Device'\033[0m"
  echo -e "\033[1;36m  3. Check:\033[0m"
  echo -e "\033[1;32m     ✅ BlackHole 16ch\033[0m"
  echo -e "\033[1;32m     ✅ Your speaker (e.g. MacBook Pro Speakers or AirPods)\033[0m"
  echo -e "\033[1;36m  4. (Optional) Enable 'Drift Correction' for speaker\033[0m"
  echo -e "\033[1;36m  5. Close the window when done.\033[0m"
  echo ""
  echo -e "\033[1;33m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"

  read -p "🕐 Press Enter when you have created the Multi-Output Device..."

  echo "🔊 Switching system output to 'Multi-Output Device'..."
  SwitchAudioSource -s "Multi-Output Device"

  echo ""
  echo "🎉 Setup complete!"
  echo "✅ You can now hear sound AND record system audio using BlackHole!"
  echo "Auto close in 3 seconds..."

  # 等待 3 秒再自动关闭（适用于 Terminal.app）
  sleep 3
  osascript -e 'tell application "Terminal" to close front window' & exit
  exit
fi