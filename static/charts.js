/**
 * Charts JavaScript
 * Handles Chart.js initialization and updates
 */

// ============================================
// Chart Instances
// ============================================
const ChartInstances = {};

// ============================================
// Initialize All Charts
// ============================================
function initializeCharts() {
    initializeAccuracyChart();
    initializeROMChart();
    initializeCompletionChart();
    initializeRecoveryChart();
}

// ============================================
// Accuracy Chart
// ============================================
function initializeAccuracyChart() {
    const canvas = document.getElementById('accuracyChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    ChartInstances.accuracy = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Accuracy',
                data: [72, 78, 75, 82, 88, 85, 91],
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                tension: 0.4,
                fill: true,
                borderWidth: 3,
                pointBackgroundColor: '#4CAF50',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(255,255,255,0.9)',
                    titleColor: '#1a2332',
                    bodyColor: '#5a6b7c',
                    borderColor: 'rgba(0,0,0,0.05)',
                    borderWidth: 1,
                    cornerRadius: 12,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `Accuracy: ${context.parsed.y}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(0,0,0,0.04)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 11
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

// ============================================
// ROM Chart
// ============================================
function initializeROMChart() {
    const canvas = document.getElementById('romChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    ChartInstances.rom = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Range of Motion',
                data: [45, 52, 48, 58, 65, 62, 72],
                backgroundColor: 'rgba(33, 150, 243, 0.7)',
                borderColor: '#2196F3',
                borderWidth: 2,
                borderRadius: 6,
                hoverBackgroundColor: 'rgba(33, 150, 243, 0.9)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(255,255,255,0.9)',
                    titleColor: '#1a2332',
                    bodyColor: '#5a6b7c',
                    borderColor: 'rgba(0,0,0,0.05)',
                    borderWidth: 1,
                    cornerRadius: 12,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `ROM: ${context.parsed.y}°`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(0,0,0,0.04)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '°';
                        },
                        font: {
                            size: 11
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Completion Chart
// ============================================
function initializeCompletionChart() {
    const canvas = document.getElementById('completionChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    ChartInstances.completion = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Completed', 'In Progress', 'Not Started'],
            datasets: [{
                data: [65, 25, 10],
                backgroundColor: [
                    '#4CAF50',
                    '#FF9800',
                    '#e0e0e0'
                ],
                borderColor: '#fff',
                borderWidth: 3,
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(255,255,255,0.9)',
                    titleColor: '#1a2332',
                    bodyColor: '#5a6b7c',
                    borderColor: 'rgba(0,0,0,0.05)',
                    borderWidth: 1,
                    cornerRadius: 12,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ${percentage}%`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Recovery Chart
// ============================================
function initializeRecoveryChart() {
    const canvas = document.getElementById('recoveryChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    ChartInstances.recovery = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
            datasets: [{
                label: 'Recovery Score',
                data: [45, 52, 61, 68, 73, 82],
                borderColor: '#9C27B0',
                backgroundColor: 'rgba(156, 39, 176, 0.1)',
                tension: 0.4,
                fill: true,
                borderWidth: 3,
                pointBackgroundColor: '#9C27B0',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(255,255,255,0.9)',
                    titleColor: '#1a2332',
                    bodyColor: '#5a6b7c',
                    borderColor: 'rgba(0,0,0,0.05)',
                    borderWidth: 1,
                    cornerRadius: 12,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `Recovery: ${context.parsed.y}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(0,0,0,0.04)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 11
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Update Charts with New Data
// ============================================
function updateCharts(period = 'week') {
    // In production, fetch new data based on period
    // For now, just show a toast
    showToast(`Updating charts for ${period}`, 'success');
    
    // Example of updating chart data
    // if (ChartInstances.accuracy) {
    //     ChartInstances.accuracy.data.datasets[0].data = newData;
    //     ChartInstances.accuracy.update();
    // }
}

// ============================================
// Radar Chart for Joint Analysis
// ============================================
function createRadarChart(canvasId, data, labels) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    
    const ctx = canvas.getContext('2d');
    
    return new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels || ['Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle'],
            datasets: [{
                label: 'Current',
                data: data || [75, 82, 68, 90, 78, 85],
                backgroundColor: 'rgba(76, 175, 80, 0.2)',
                borderColor: '#4CAF50',
                borderWidth: 2,
                pointBackgroundColor: '#4CAF50',
                pointRadius: 4
            }, {
                label: 'Previous',
                data: [65, 72, 58, 80, 68, 75],
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                borderColor: '#2196F3',
                borderWidth: 2,
                pointBackgroundColor: '#2196F3',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.06)'
                    }
                }
            }
        }
    });
}

// ============================================
// Export for use in other files
// ============================================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeCharts,
        updateCharts,
        createRadarChart,
        ChartInstances
    };
}