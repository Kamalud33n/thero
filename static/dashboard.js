/**
 * Dashboard JavaScript
 * Handles dashboard-specific functionality
 */

// ============================================
// Dashboard State
// ============================================
const DashboardState = {
    currentPeriod: 'week',
    chartInstances: {},
    statsData: null,
    progressData: null
};

// ============================================
// Initialize Dashboard
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    if (document.querySelector('.content-area')) {
        initializeDashboard();
    }
});

function initializeDashboard() {
    loadDashboardStats();
    loadProgressTimeline();
    setupChartPeriodButtons();
    setupFilterButtons();
}

// ============================================
// Load Dashboard Statistics
// ============================================
async function loadDashboardStats() {
    try {
        const response = await fetch('/api/dashboard');
        const data = await response.json();
        DashboardState.statsData = data;
        
        updateStatsCards(data.statistics);
        updateRecentActivity(data.progress);
    } catch (error) {
        console.error('Error loading dashboard stats:', error);
        showToast('Error loading dashboard data', 'error');
    }
}

function updateStatsCards(stats) {
    const statsGrid = document.getElementById('dashboardStats');
    if (!statsGrid) return;
    
    const cards = [
        {
            icon: 'fa-calendar-check',
            label: 'Today\'s Sessions',
            value: stats.today_sessions,
            trend: '+12%',
            trendUp: true
        },
        {
            icon: 'fa-check-circle',
            label: 'Exercises Completed',
            value: stats.completed_exercises,
            trend: '+8%',
            trendUp: true
        },
        {
            icon: 'fa-percent',
            label: 'Avg Accuracy',
            value: stats.avg_accuracy + '%',
            trend: '+5%',
            trendUp: true
        },
        {
            icon: 'fa-arrow-right',
            label: 'Avg ROM',
            value: stats.avg_rom + '°',
            trend: '+3%',
            trendUp: true
        },
        {
            icon: 'fa-heart',
            label: 'Recovery Score',
            value: stats.recovery_score + '%',
            trend: '+15%',
            trendUp: true
        },
        {
            icon: 'fa-shield-alt',
            label: 'Avg Stability',
            value: stats.avg_stability + '%',
            trend: '+6%',
            trendUp: true
        },
        {
            icon: 'fa-balance-scale',
            label: 'Avg Balance',
            value: stats.avg_balance + '%',
            trend: '+4%',
            trendUp: true
        },
        {
            icon: 'fa-chart-line',
            label: 'Improvement',
            value: '+12.5%',
            trend: '+12.5%',
            trendUp: true
        }
    ];
    
    statsGrid.innerHTML = cards.map((card, index) => `
        <div class="stat-card glass-effect animate-slide-up" style="animation-delay: ${index * 0.05}s">
            <div class="stat-icon" style="background: ${getColorForStat(index)}20; color: ${getColorForStat(index)}">
                <i class="fas ${card.icon}"></i>
            </div>
            <div class="stat-content">
                <span class="stat-value">${card.value}</span>
                <span class="stat-label">${card.label}</span>
            </div>
            <div class="stat-trend ${card.trendUp ? '' : 'down'}">
                <i class="fas fa-${card.trendUp ? 'arrow-up' : 'arrow-down'}"></i>
                <span>${card.trend}</span>
            </div>
        </div>
    `).join('');
}

function getColorForStat(index) {
    const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#f44336', '#00BCD4', '#4CAF50', '#FF9800'];
    return colors[index % colors.length];
}

// ============================================
// Recent Activity
// ============================================
function updateRecentActivity(progress) {
    const activityList = document.getElementById('recentActivity');
    if (!activityList) return;
    
    if (!progress || progress.length === 0) {
        activityList.innerHTML = `
            <div class="activity-empty">
                <i class="fas fa-inbox"></i>
                <span>No recent sessions</span>
                <span class="empty-sub">Start a session to track progress</span>
            </div>
        `;
        return;
    }
    
    activityList.innerHTML = progress.slice(0, 5).map(item => `
        <div class="activity-item animate-slide-up">
            <div class="activity-icon ${item.accuracy > 80 ? 'success' : 'warning'}">
                <i class="fas fa-${item.accuracy > 80 ? 'check-circle' : 'exclamation-circle'}"></i>
            </div>
            <div class="activity-details">
                <div class="activity-title">${item.exercise || 'Exercise Session'}</div>
                <div class="activity-meta">
                    <span>${item.date || 'Today'}</span>
                    <span>•</span>
                    <span>Accuracy: ${(item.accuracy || 0).toFixed(1)}%</span>
                    <span>•</span>
                    <span>ROM: ${(item.rom || 0).toFixed(1)}°</span>
                </div>
            </div>
            <div class="activity-status">
                <span class="status-badge ${item.accuracy > 80 ? 'success' : 'warning'}">
                    ${item.accuracy > 80 ? 'Excellent' : 'Needs Improvement'}
                </span>
            </div>
        </div>
    `).join('');
}

// ============================================
// Progress Timeline
// ============================================
async function loadProgressTimeline() {
    try {
        const response = await fetch('/api/dashboard');
        const data = await response.json();
        
        const timeline = document.getElementById('progressTimeline');
        if (!timeline) return;
        
        if (data.progress && data.progress.length > 0) {
            timeline.innerHTML = data.progress.map(item => `
                <div class="timeline-item">
                    <div class="timeline-date">${item.date}</div>
                    <div class="timeline-bars">
                        <div class="timeline-bar" style="width: ${item.accuracy}%; background: #4CAF50;">
                            <span>Accuracy: ${item.accuracy.toFixed(1)}%</span>
                        </div>
                        <div class="timeline-bar" style="width: ${(item.rom / 100) * 100}%; background: #2196F3;">
                            <span>ROM: ${item.rom.toFixed(1)}°</span>
                        </div>
                    </div>
                </div>
            `).join('');
        } else {
            timeline.innerHTML = `
                <div class="activity-empty">
                    <i class="fas fa-chart-line"></i>
                    <span>No progress data available</span>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading progress timeline:', error);
    }
}

// ============================================
// Chart Period Buttons
// ============================================
function setupChartPeriodButtons() {
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const container = this.closest('.chart-container');
            if (container) {
                container.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                const period = this.dataset.period;
                updateChartPeriod(period);
            }
        });
    });
}

function updateChartPeriod(period) {
    DashboardState.currentPeriod = period;
    // Update charts with new period
    if (window.updateCharts) {
        window.updateCharts(period);
    }
}

// ============================================
// Filter Buttons
// ============================================
function setupFilterButtons() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const container = this.closest('.progress-summary');
            if (container) {
                container.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                const filter = this.textContent.toLowerCase();
                applyProgressFilter(filter);
            }
        });
    });
}

function applyProgressFilter(filter) {
    const items = document.querySelectorAll('.timeline-item');
    items.forEach(item => {
        if (filter === 'all') {
            item.style.display = 'block';
        } else {
            // Filter logic based on filter type
            item.style.display = 'block'; // Simplified
        }
    });
}

// ============================================
// Period Selection
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    const periodSelect = document.getElementById('reportPeriod');
    if (periodSelect) {
        periodSelect.addEventListener('change', function() {
            updateDashboardPeriod(this.value);
        });
    }
});

function updateDashboardPeriod(period) {
    loadDashboardStats();
    loadProgressTimeline();
    showToast(`Viewing ${period} data`, 'success');
}