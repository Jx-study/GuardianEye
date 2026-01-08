// ========== 全域變數 ==========
let isPasswordVisible = false;
let updateInterval = null;

// ========== 頁面載入時啟動 ==========
$(document).ready(function () {
    console.log('🚀 Guardian Eye 前端啟動');

    // 綁定事件
    bindEvents();

    // 開始輪詢
    startAutoUpdate();

    // 載入歷史記錄
    loadHistory();
});

// ========== 事件綁定 ==========
function bindEvents() {
    // Tab 切換
    $('.tab-btn').on('click', function () {
        const tabName = $(this).data('tab');
        switchTab(tabName);
    });

    // 密碼顯示/隱藏
    $('#toggle-password').on('click', togglePasswordVisibility);

    // 解鎖按鈕
    $('#unlock-btn').on('click', handleUnlock);

    // 密碼輸入框 Enter 鍵
    $('#password-input').on('keypress', function (e) {
        if (e.key === 'Enter') {
            handleUnlock();
        }
    });

    // 控制按鈕
    $('.btn-control').on('click', function () {
        const action = $(this).data('action');
        handleControl(action);
    });

    // 刷新歷史記錄
    $('#refresh-history').on('click', loadHistory);
}

// ========== Tab 切換 ==========
function switchTab(tabName) {
    // 移除所有 active 狀態
    $('.tab-btn').removeClass('active');
    $('.tab-pane').removeClass('active');

    // 添加 active 狀態到選中的 tab
    $(`[data-tab="${tabName}"]`).addClass('active');
    $(`#tab-${tabName}`).addClass('active');
}

// ========== 啟動自動更新 ==========
function startAutoUpdate() {
    if (updateInterval) clearInterval(updateInterval);
    updateInterval = setInterval(updateStatus, 1000);
    updateStatus(); // 立即執行一次
}

// ========== 更新狀態 (AJAX) ==========
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        // 更新連線狀態
        updateConnectionStatus(true);

        // 更新警報狀態
        updateAlarmStatus(data);

        // 更新解鎖狀態
        updateUnlockStatus(data);

        // 更新系統資訊
        updateSystemInfo(data);

    } catch (error) {
        console.error('❌ 更新狀態失敗:', error);
        updateConnectionStatus(false);
    }
}

// ========== 更新連線狀態 ==========
function updateConnectionStatus(isOnline) {
    if (isOnline) {
        $('#connection-status').removeClass('offline').addClass('online');
    } else {
        $('#connection-status').removeClass('online').addClass('offline');
    }
}

// ========== 更新警報狀態 ==========
function updateAlarmStatus(data) {
    if (data.alarm_active) {
        // 顯示警報
        $('#alarm-status-normal').hide();
        $('#alarm-status-active').show();

        // 更新警報類型
        const alarmTypes = {
            'pig': '偵測到小豬玩偶入侵！',
            'stranger': '偵測到陌生人！',
            'default': '偵測到異常活動！'
        };

        $('#alarm-type-text').text(alarmTypes[data.alarm_type] || alarmTypes.default);
        $('#alarm-time-text').text(`觸發時間：${data.alarm_time || '--'}`);
        $('#alarm-confidence-text').text(`信心度：${(data.alarm_confidence * 100).toFixed(1)}%`);

    } else {
        // 系統正常
        $('#alarm-status-normal').show();
        $('#alarm-status-active').hide();
    }
}

// ========== 更新解鎖狀態 ==========
function updateUnlockStatus(data) {
    if (data.remote_unlocked) {
        // 遠端已解鎖
        $('#unlock-form').hide();
        $('#unlock-success').show();
    } else {
        // 尚未解鎖
        $('#unlock-form').show();
        $('#unlock-success').hide();
    }
}

// ========== 更新系統資訊 ==========
function updateSystemInfo(data) {
    // 伺服器時間
    if (data.server_time) {
        const time = new Date(data.server_time);
        $('#server-time').text(time.toLocaleTimeString('zh-TW'));
    }

    // 運行時間
    if (data.uptime !== undefined) {
        $('#uptime').text(formatUptime(data.uptime));
    }

    // 最後更新時間
    $('#last-update').text(new Date().toLocaleTimeString('zh-TW'));
}

// ========== 格式化運行時間 ==========
function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (days > 0) {
        return `${days}天 ${hours}小時`;
    } else if (hours > 0) {
        return `${hours}小時 ${minutes}分`;
    } else if (minutes > 0) {
        return `${minutes}分 ${secs}秒`;
    } else {
        return `${secs}秒`;
    }
}

// ========== 密碼顯示/隱藏 ==========
function togglePasswordVisibility() {
    isPasswordVisible = !isPasswordVisible;

    if (isPasswordVisible) {
        $('#password-input').attr('type', 'text');
        $('#toggle-password').html('<i class="fas fa-eye-slash"></i>');
    } else {
        $('#password-input').attr('type', 'password');
        $('#toggle-password').html('<i class="fas fa-eye"></i>');
    }
}

// ========== 處理解鎖 (AJAX) ==========
async function handleUnlock() {
    const password = $('#password-input').val().trim();

    // 驗證輸入
    if (!password) {
        showMessage('#unlock-message', '請輸入密碼', 'error');
        return;
    }

    // 禁用按鈕
    $('#unlock-btn').prop('disabled', true)
        .html('<i class="fas fa-spinner fa-spin"></i> 驗證中...');

    try {
        const response = await fetch('/api/unlock', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: password })
        });

        const data = await response.json();

        if (data.success) {
            // 解鎖成功
            showMessage('#unlock-message', data.message, 'success');
            $('#password-input').val('');

            // 2秒後隱藏訊息
            setTimeout(() => {
                $('#unlock-message').removeClass('show');
            }, 2000);

        } else {
            // 解鎖失敗
            showMessage('#unlock-message', data.message, 'error');
        }

    } catch (error) {
        console.error('❌ 解鎖請求失敗:', error);
        showMessage('#unlock-message', '❌ 網路錯誤，請稍後再試', 'error');
    } finally {
        // 恢復按鈕
        $('#unlock-btn').prop('disabled', false)
            .html('<i class="fas fa-unlock"></i> 解鎖');
    }
}

// ========== 處理控制指令 (AJAX) ==========
async function handleControl(action) {
    const actionNames = {
        'mute_alarm': '靜音警報',
        'open_door': '強制開門',
        'reset': '重置系統',
        'test_alarm': '測試警報'
    };

    const actionName = actionNames[action] || action;

    // 確認操作
    if (!confirm(`確定要執行「${actionName}」嗎？`)) {
        return;
    }

    try {
        const response = await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action })
        });

        const data = await response.json();

        if (data.success) {
            showMessage('#control-message', data.message, 'success');
        } else {
            showMessage('#control-message', data.message, 'error');
        }

    } catch (error) {
        console.error('❌ 控制指令失敗:', error);
        showMessage('#control-message', '❌ 網路錯誤，請稍後再試', 'error');
    }
}

// ========== 載入歷史記錄 (AJAX) ==========
async function loadHistory() {
    try {
        $('#history-list').html('<p class="loading">載入中...</p>');

        const response = await fetch('/api/history');
        const logs = await response.json();

        if (logs.length === 0) {
            $('#history-list').html('<p class="loading">無歷史記錄</p>');
            return;
        }

        // 渲染歷史記錄
        const historyHTML = logs.map(log => `
            <div class="history-item fade-in">
                <div class="history-info">
                    <div class="history-time">${log.time}</div>
                    <div class="history-event">${log.event}</div>
                </div>
                <div class="history-status ${getStatusClass(log.status)}">
                    ${log.status}
                </div>
            </div>
        `).join('');

        $('#history-list').html(historyHTML);

    } catch (error) {
        console.error('❌ 載入歷史記錄失敗:', error);
        $('#history-list').html('<p class="loading">載入失敗</p>');
    }
}

// ========== 取得狀態樣式 ==========
function getStatusClass(status) {
    if (status.includes('成功') || status.includes('正常') || status.includes('已解除')) {
        return 'success';
    } else if (status.includes('失敗') || status.includes('錯誤')) {
        return 'failed';
    } else {
        return 'normal';
    }
}

// ========== 顯示訊息 ==========
function showMessage(selector, message, type) {
    $(selector)
        .text(message)
        .removeClass('success error info')
        .addClass(`show ${type}`);

    // 3秒後自動隱藏
    setTimeout(() => {
        $(selector).removeClass('show');
    }, 3000);
}

// ========== 清理 ==========
$(window).on('beforeunload', function () {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});
