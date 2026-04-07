let imagePage = 1;
let imageTotalPages = 1;

async function api(url, options={}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function switchPage(page) {
  document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.page === page));
  document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
}

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => switchPage(btn.dataset.page));
});

async function loadConfig() {
  const data = await api('/api/config');
  document.getElementById('datasetRootInput').value = data.dataset_root;
}

async function saveConfig() {
  const dataset_root = document.getElementById('datasetRootInput').value.trim();
  await api('/api/config', { method: 'POST', body: JSON.stringify({ dataset_root }) });
  await refreshAll();
  alert('Dataset path updated');
}

function renderJobs(targetId, jobs) {
  const target = document.getElementById(targetId);
  if (!jobs.length) {
    target.innerHTML = '<div class="muted">No jobs yet.</div>';
    return;
  }
  target.innerHTML = jobs.map(job => `
    <div class="job-item">
      <div><strong>${job.model}</strong> on <strong>${job.split}</strong></div>
      <div>${job.count} images | dim ${job.feature_dim} | ${job.duration_seconds}s</div>
      <div class="muted">${job.finished_at}</div>
    </div>
  `).join('');
}

async function loadDashboard() {
  const data = await api('/api/dashboard');
  document.getElementById('totalImages').textContent = data.total_images;
  document.getElementById('splitCounts').textContent = `Train: ${data.train_count} | Query: ${data.query_count} | Test: ${data.test_count}`;
  document.getElementById('totalVehicles').textContent = data.total_vehicles;
  document.getElementById('totalCameras').textContent = data.total_cameras;
  document.getElementById('featurePercent').textContent = `${data.feature_percent}%`;
  document.getElementById('featureCount').textContent = `${data.features_extracted} / ${data.total_images} images`;
  renderJobs('recentJobs', data.jobs || []);
}

async function loadImages() {
  const split = document.getElementById('imgSplit').value;
  const vehicle_id = document.getElementById('vehicleId').value.trim();
  const camera_id = document.getElementById('cameraId').value.trim();
  const has_features = document.getElementById('hasFeatures').value;

  const params = new URLSearchParams({ split, vehicle_id, camera_id, has_features, page: String(imagePage), page_size: '24' });
  const data = await api(`/api/images?${params.toString()}`);

  imageTotalPages = data.total_pages || 1;
  document.getElementById('imagesFound').textContent = `Found ${data.total} images`;
  document.getElementById('pageInfo').textContent = `Page ${data.page} of ${data.total_pages}`;

  const grid = document.getElementById('imageGrid');
  if (!data.items.length) {
    grid.innerHTML = '<div class="muted">No images found.</div>';
    return;
  }

  grid.innerHTML = data.items.map(item => `
    <div class="img-card">
      <img src="${item.image_url}" alt="${item.filename}">
      <div class="img-meta">
        <div><strong>${item.filename}</strong></div>
        <div>Vehicle: ${item.vehicle_id}</div>
        <div>Camera: ${item.camera_id}</div>
        <div>Features: ${item.has_features ? 'Yes' : 'No'}</div>
      </div>
    </div>
  `).join('');
}

async function startExtraction() {
  const model = document.querySelector('input[name="model"]:checked').value;
  const split = document.querySelector('input[name="target_split"]:checked').value;
  const batch_size = Number(document.getElementById('batchSize').value || 32);
  const custom_ids = document.getElementById('customIds').value
    .split(',')
    .map(x => x.trim())
    .filter(Boolean);

  const resultEl = document.getElementById('extractResult');
  resultEl.textContent = 'Running extraction...';

  try {
    const data = await api('/api/extract', {
      method: 'POST',
      body: JSON.stringify({ model, split, batch_size, custom_ids })
    });
    resultEl.textContent = JSON.stringify(data, null, 2);
    await refreshAll();
  } catch (err) {
    resultEl.textContent = err.message;
  }
}

async function loadJobs() {
  const data = await api('/api/jobs');
  renderJobs('jobsList', data.jobs || []);
}

async function refreshAll() {
  await Promise.all([loadConfig(), loadDashboard(), loadImages(), loadJobs()]);
}

document.getElementById('saveRootBtn').addEventListener('click', saveConfig);
document.getElementById('refreshDashboard').addEventListener('click', loadDashboard);
document.getElementById('refreshImages').addEventListener('click', loadImages);
document.getElementById('applyFilters').addEventListener('click', () => { imagePage = 1; loadImages(); });
document.getElementById('refreshJobs').addEventListener('click', loadJobs);
document.getElementById('startExtraction').addEventListener('click', startExtraction);
document.getElementById('prevPage').addEventListener('click', () => { if (imagePage > 1) { imagePage--; loadImages(); } });
document.getElementById('nextPage').addEventListener('click', () => { if (imagePage < imageTotalPages) { imagePage++; loadImages(); } });

refreshAll().catch(err => {
  console.error(err);
  alert(err.message);
});
