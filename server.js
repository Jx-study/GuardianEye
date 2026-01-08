// server.js
const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 8080;

// 中介軟體
app.use(express.json());
app.use(express.static("public"));  // 提供靜態檔案（HTML/CSS/JS）

// ========================= 共享檔案路徑 =========================
const ALARM_STATUS_FILE = "/tmp/guardian_alarm_status.json";
const UNLOCK_STATUS_FILE = "/tmp/guardian_unlock_status.json";
const CONTROL_FILE = "/tmp/guardian_control.txt";

// 預設密碼（可以改成從環境變數讀取）
const CORRECT_PASSWORD = process.env.UNLOCK_PASSWORD || "1234";

// ========================= 工具函數 =========================

/**
 * 讀取警報狀態（Qt 寫入，Node.js 讀取）
 */
function readAlarmStatus() {
  try {
    if (fs.existsSync(ALARM_STATUS_FILE)) {
      const data = fs.readFileSync(ALARM_STATUS_FILE, "utf8");
      return JSON.parse(data);
    }
  } catch (error) {
    console.error("讀取警報狀態失敗:", error.message);
  }

  // 預設值
  return {
    alarm_active: false,
    alarm_type: null,       // 'pig', 'stranger', null
    timestamp: null,
    image_path: null,
    confidence: 0
  };
}

/**
 * 讀取解鎖狀態（Qt 讀取，Node.js 寫入）
 */
function readUnlockStatus() {
  try {
    if (fs.existsSync(UNLOCK_STATUS_FILE)) {
      const data = fs.readFileSync(UNLOCK_STATUS_FILE, "utf8");
      return JSON.parse(data);
    }
  } catch (error) {
    console.error("讀取解鎖狀態失敗:", error.message);
  }

  return {
    remote_unlocked: false,
    password_correct: false,
    timestamp: null,
    random_code: null
  };
}

/**
 * 寫入解鎖狀態（供 Qt 輪詢）
 */
function writeUnlockStatus(data) {
  try {
    fs.writeFileSync(UNLOCK_STATUS_FILE, JSON.stringify(data, null, 2));
    console.log("✅ 解鎖狀態已更新:", data);
  } catch (error) {
    console.error("寫入解鎖狀態失敗:", error.message);
  }
}

/**
 * 讀取系統日誌（從 Kernel Driver 或檔案）
 */
function readSystemLogs() {
  // 這裡可以從 Kernel Driver 讀取，或從日誌檔讀取
  // 簡化版：從檔案讀取
  const LOG_FILE = "/tmp/guardian_logs.json";

  try {
    if (fs.existsSync(LOG_FILE)) {
      const data = fs.readFileSync(LOG_FILE, "utf8");
      return JSON.parse(data);
    }
  } catch (error) {
    console.error("讀取日誌失敗:", error.message);
  }

  // 預設假資料（展示用）
  return [
    {
      time: "2025-01-09 14:25:05",
      event: "小豬入侵警報",
      status: "已解除",
      type: "alarm"
    },
    {
      time: "2025-01-09 10:15:32",
      event: "陌生人偵測",
      status: "已通知",
      type: "detection"
    },
    {
      time: "2025-01-09 08:30:15",
      event: "主人開門",
      status: "正常",
      type: "access"
    }
  ];
}

// ========================= API 路由 =========================

/**
 * GET /api/status
 * 功能：取得當前系統狀態（警報、解鎖狀態）
 * 前端會每秒輪詢這個 API
 */
app.get("/api/status", (req, res) => {
  const alarmStatus = readAlarmStatus();
  const unlockStatus = readUnlockStatus();

  res.json({
    // 警報狀態
    alarm_active: alarmStatus.alarm_active,
    alarm_type: alarmStatus.alarm_type,
    alarm_time: alarmStatus.timestamp,
    alarm_confidence: alarmStatus.confidence,

    // 解鎖狀態
    remote_unlocked: unlockStatus.remote_unlocked,
    awaiting_local_code: unlockStatus.remote_unlocked && !unlockStatus.password_correct,

    // 系統資訊
    server_time: new Date().toISOString(),
    uptime: process.uptime()
  });
});

/**
 * POST /api/unlock
 * 功能：遠端解鎖（驗證密碼）
 * 請求體：{ "password": "1234" }
 */
app.post("/api/unlock", (req, res) => {
  const { password } = req.body;

  // 驗證密碼
  if (!password) {
    return res.status(400).json({
      success: false,
      message: "❌ 請輸入密碼"
    });
  }

  if (password === CORRECT_PASSWORD) {
    // 密碼正確，寫入解鎖狀態
    const unlockData = {
      remote_unlocked: true,
      password_correct: true,
      timestamp: new Date().toISOString(),
      unlock_method: "web"
    };

    writeUnlockStatus(unlockData);

    // 記錄到日誌
    logEvent("遠端解鎖成功", "unlock", "success");

    res.json({
      success: true,
      message: "✅ 遠端驗證通過！請回家輸入現場隨機密碼。"
    });
  } else {
    // 密碼錯誤
    logEvent("遠端解鎖失敗（密碼錯誤）", "unlock", "failed");

    res.status(401).json({
      success: false,
      message: "❌ 密碼錯誤，請重試。"
    });
  }
});

/**
 * GET /api/history
 * 功能：取得歷史記錄
 */
app.get("/api/history", (req, res) => {
  const logs = readSystemLogs();
  res.json(logs);
});

/**
 * POST /api/control
 * 功能：手動控制（如：強制開門、靜音警報）
 * 請求體：{ "action": "open_door" | "mute_alarm" | "reset" }
 */
app.post("/api/control", (req, res) => {
  const { action } = req.body;

  const validActions = ["open_door", "mute_alarm", "reset", "test_alarm"];

  if (!validActions.includes(action)) {
    return res.status(400).json({
      success: false,
      message: "❌ 無效的控制指令"
    });
  }

  // 寫入控制檔（Qt 輪詢讀取）
  try {
    fs.writeFileSync(CONTROL_FILE, action);
    console.log(`✅ 控制指令已發送: ${action}`);

    res.json({
      success: true,
      action: action,
      message: `✅ 已執行: ${action}`
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "❌ 控制指令發送失敗"
    });
  }
});

/**
 * GET /api/image/:filename
 * 功能：取得警報時拍攝的照片
 */
app.get("/api/image/:filename", (req, res) => {
  const filename = req.params.filename;
  const imagePath = path.join("/tmp/guardian_images", filename);

  if (fs.existsSync(imagePath)) {
    res.sendFile(imagePath);
  } else {
    res.status(404).send("圖片不存在");
  }
});

/**
 * POST /api/reset-unlock
 * 功能：重置解鎖狀態（測試用）
 */
app.post("/api/reset-unlock", (req, res) => {
  writeUnlockStatus({
    remote_unlocked: false,
    password_correct: false,
    timestamp: null
  });

  res.json({
    success: true,
    message: "✅ 解鎖狀態已重置"
  });
});

// ========================= 日誌記錄 =========================

function logEvent(eventName, eventType, status) {
  const LOG_FILE = "/tmp/guardian_logs.json";

  const newLog = {
    time: new Date().toLocaleString("zh-TW"),
    event: eventName,
    status: status,
    type: eventType
  };

  try {
    let logs = [];
    if (fs.existsSync(LOG_FILE)) {
      const data = fs.readFileSync(LOG_FILE, "utf8");
      logs = JSON.parse(data);
    }

    // 新增到最前面（最新的在上面）
    logs.unshift(newLog);

    // 只保留最近 50 筆
    if (logs.length > 50) {
      logs = logs.slice(0, 50);
    }

    fs.writeFileSync(LOG_FILE, JSON.stringify(logs, null, 2));
  } catch (error) {
    console.error("寫入日誌失敗:", error.message);
  }
}

// ========================= 健康檢查 =========================

app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

// ========================= 啟動伺服器 =========================

app.listen(PORT, "0.0.0.0", () => {
  console.log("========================================");
  console.log("🛡️  Guardian Eye Remote Control Server");
  console.log("========================================");
  console.log(`✅ Server running on http://0.0.0.0:${PORT}`);
  console.log(`📱 手機訪問（請確保在同一 WiFi）：`);

  // 嘗試取得本機 IP
  const os = require("os");
  const interfaces = os.networkInterfaces();

  Object.keys(interfaces).forEach((ifname) => {
    interfaces[ifname].forEach((iface) => {
      if (iface.family === "IPv4" && !iface.internal) {
        console.log(`   http://${iface.address}:${PORT}`);
      }
    });
  });

  console.log("========================================");
  console.log(`📂 靜態檔案目錄: ${path.join(__dirname, "public")}`);
  console.log(`🔐 解鎖密碼: ${CORRECT_PASSWORD}`);
  console.log("========================================");
});

// 優雅關閉
process.on("SIGINT", () => {
  console.log("\n\n👋 伺服器正在關閉...");
  process.exit(0);
});
