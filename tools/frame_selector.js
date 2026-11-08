/* SPDX-License-Identifier: Apache-2.0 */
'use strict';
(() => {
  const D = JSON.parse(document.getElementById('selector-data').textContent);
  const $ = id => document.getElementById(id);
  const clone = value => JSON.parse(JSON.stringify(value));
  const esc = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  } [c]));
  const scenes = new Map(D.scenes.map(s => [s.scene_id, s]));
  const frames = new Map(D.scenes.flatMap(s => s.frames.map(f => [f.clip_id, {
    scene: s,
    frame: f
  }])));
  const allKeys = Object.entries(D.taxonomy).flatMap(([axis, values]) => Object.keys(values).map(v => axis + ':' + v));
  const labelFor = key => {
    const [axis, value] = key.split(':');
    return D.taxonomy[axis][value];
  };
  const keys = c => ['longitudinal:' + c.longitudinal, 'lateral:' + c.lateral, ...c.causal_factors.map(v => 'causal_factors:' + v)];

  function validCategories(c) {
    if (!c || Object.keys(c).sort().join(',') !== 'causal_factors,lateral,longitudinal') throw Error('Invalid category object');
    for (const a of ['longitudinal', 'lateral'])
      if (!Object.hasOwn(D.taxonomy[a], c[a])) throw Error('Unknown ' + a + ' category');
    if (!Array.isArray(c.causal_factors) || new Set(c.causal_factors).size !== c.causal_factors.length || c.causal_factors.some(v => !Object.hasOwn(D.taxonomy.causal_factors, v))) throw Error('Invalid causal factors');
    return c;
  }
  const storageKey = 'frame-selector-v1:' + D.dataset_id;
  let state = {
    schema_version: 1,
    dataset_id: D.dataset_id,
    selector: '',
    targets: Object.fromEntries(allKeys.map(k => [k, k.endsWith(':none') ? 0 : D.target_n])),
    selected: {},
    suggestions: {},
    events: []
  };
  let currentScene = D.scenes.find(s => s.frames.length)?.scene_id || D.scenes[0].scene_id;
  let frameIndex = 0,
    focus = '',
    search = '';

  function notify(message, error = false) {
    $('status').textContent = message;
    $('status').className = error ? 'error' : 'muted';
  }

  function validateState(raw) {
    if (raw.schema_version !== 1 || raw.dataset_id !== D.dataset_id) throw Error('This session belongs to a different candidate dataset or benchmark. Keep the original workspace or reconcile the selection before importing.');
    if (!raw.targets || !raw.selected || !raw.suggestions || !Array.isArray(raw.events) || typeof raw.selector !== 'string') throw Error('Incomplete session');
    if (Object.keys(raw.targets).sort().join('|') !== [...allKeys].sort().join('|')) throw Error('Invalid target categories');
    for (const n of Object.values(raw.targets))
      if (!Number.isInteger(n) || n < 0) throw Error('Targets must be nonnegative integers');
    for (const [sid, item] of Object.entries(raw.selected)) {
      const found = frames.get(item.clip_id);
      if (!found || found.scene.scene_id !== sid) throw Error('Selection refers to an unknown, excluded or wrong-scene frame');
      validCategories(item.categories);
      if (typeof item.reason !== 'string' || !item.reason.trim() || !item.confirmed) throw Error('Selection needs a reason and confirmed categories');
    }
    const cotIds = new Set([...frames.values()].map(x => x.frame.cot_id));
    for (const [id, item] of Object.entries(raw.suggestions)) {
      if (!cotIds.has(id) || typeof item.source !== 'string' || !item.source.trim()) throw Error('Unknown CoT or missing suggestion source');
      validCategories(item.categories);
    }
    return clone(raw);
  }
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      state = validateState(JSON.parse(saved));
      notify('Restored browser session.');
    }
  } catch (e) {
    notify('Autosave could not be restored: ' + e.message, true);
  }

  function save() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
    } catch (e) {
      notify('Browser autosave failed. Download a session backup: ' + e.message, true);
    }
  }

  function event(action, details) {
    state.events.push({
      action,
      at: new Date().toISOString(),
      selector: state.selector,
      ...details
    });
    save();
  }

  function proposal(frame) {
    return state.suggestions[frame.cot_id] || {
      categories: frame.categories,
      source: frame.category_source
    };
  }

  function counts(exceptScene = null) {
    const n = {
      ...D.baseline_counts
    };
    for (const [sid, item] of Object.entries(state.selected))
      if (sid !== exceptScene)
        for (const k of keys(item.categories)) n[k] = (n[k] || 0) + 1;
    return n;
  }

  function gain(categories, n) {
    if (!categories) return 0;
    return keys(categories).reduce((sum, k) => sum + ((n[k] || 0) < state.targets[k] ? (state.targets[k] - (n[k] || 0)) / Math.max(1, state.targets[k]) : 0), 0);
  }

  function availableScene(s, key) {
    return s.frames.some(f => proposal(f).categories && keys(proposal(f).categories).includes(key));
  }

  function renderDashboard() {
    const n = counts();
    const known = [...frames.values()].filter(x => proposal(x.frame).categories).length;
    $('summary').textContent = `${D.baseline_cases} existing cases · ${Object.keys(state.selected).length} chosen clips · ${known}/${frames.size} candidate decisions have provisional category tags · ${D.baseline_missing_categories.length} existing cases lack category tags`;
    $('dashboard').innerHTML = Object.entries(D.taxonomy).map(([axis, values]) => {
      const title = {
        longitudinal: 'Longitudinal decision',
        lateral: 'Lateral decision',
        causal_factors: 'Causal factor'
      } [axis];
      return `<div><h3>${title}</h3><table class="counts"><thead><tr><th>Category</th><th>Existing</th><th>Added</th><th>Target</th><th>Gap</th><th>Avail.</th></tr></thead><tbody>` + Object.entries(values).map(([v, label]) => {
        const k = axis + ':' + v,
          base = D.baseline_counts[k] || 0,
          added = (n[k] || 0) - base,
          target = state.targets[k],
          scale = Math.max(1, target, base + added);
        const avail = D.scenes.filter(s => !state.selected[s.scene_id] && availableScene(s, k)).length;
        return `<tr><td><button data-focus="${esc(k)}">${esc(label)}</button><div class="bar"><span class="base" style="width:${100*base/scale}%"></span><span class="added" style="width:${100*added/scale}%"></span></div></td><td>${base}</td><td>${added}</td><td><input aria-label="Target ${esc(label)}" type="number" min="0" step="1" value="${target}" data-target="${esc(k)}"></td><td class="deficit">${Math.max(0,target-base-added)}</td><td>${avail}</td></tr>`;
      }).join('') + '</tbody></table></div>';
    }).join('');
    document.querySelectorAll('[data-focus]').forEach(b => b.onclick = () => {
      focus = b.dataset.focus;
      $('focus').value = focus;
      renderLists();
    });
    document.querySelectorAll('[data-target]').forEach(input => input.onchange = () => {
      const value = Number(input.value);
      if (!Number.isInteger(value) || value < 0) {
        notify('Target must be a nonnegative integer.', true);
        renderDashboard();
        return;
      }
      const key = input.dataset.target,
        previous = state.targets[key];
      state.targets[key] = value;
      event('target_changed', {
        key,
        previous,
        value
      });
      renderDashboard();
      renderLists();
      renderImpact();
    });
  }

  function visibleScene(s) {
    if ($('only-selected').checked && !state.selected[s.scene_id]) return false;
    if (focus && !availableScene(s, focus) && !(state.selected[s.scene_id] && keys(state.selected[s.scene_id].categories).includes(focus))) return false;
    return !search || s.name.toLowerCase().includes(search) || s.frames.some(f => f.chain_of_thought.toLowerCase().includes(search));
  }

  function sceneScore(s, n) {
    return Math.max(0, ...s.frames.map(f => gain(proposal(f).categories, n)));
  }

  function renderLists() {
    const n = counts();
    const ordered = D.scenes.filter(visibleScene).sort((a, b) => sceneScore(b, n) - sceneScore(a, n) || a.name.localeCompare(b.name));
    $('scenes').innerHTML = ordered.map(s => `<button class="scene-button ${s.scene_id===currentScene?'active':''} ${state.selected[s.scene_id]?'selected':''}" data-scene="${esc(s.scene_id)}">${esc(s.name.split('_clipgt')[0])}<span>${esc(s.scene_id)}</span><span>${state.selected[s.scene_id]?'Selected · ':''}${s.frames.length} decisions · ${esc(s.status)}</span></button>`).join('') || '<p>No clips match this filter.</p>';
    document.querySelectorAll('[data-scene]').forEach(b => b.onclick = () => openScene(b.dataset.scene));
    renderDecisions();
    renderSelected();
  }

  function renderDecisions() {
    const s = scenes.get(currentScene),
      n = counts(currentScene);
    const candidates = s.frames.map((f, i) => ({
      f,
      i
    })).filter(({
      f
    }) => (!focus || proposal(f).categories && keys(proposal(f).categories).includes(focus)) && (!search || f.chain_of_thought.toLowerCase().includes(search) || s.name.toLowerCase().includes(search)));
    candidates.sort((a, b) => gain(proposal(b.f).categories, n) - gain(proposal(a.f).categories, n) || a.i - b.i);
    $('decisions').innerHTML = candidates.map(({
      f,
      i
    }) => `<button data-frame="${i}">${f.video_time_s.toFixed(1)}s · ${esc(f.chain_of_thought)}<span class="muted"> · priority ${gain(proposal(f).categories,n).toFixed(2)}</span></button>`).join('');
    document.querySelectorAll('[data-frame]').forEach(b => b.onclick = () => openFrame(Number(b.dataset.frame)));
  }

  function renderSelected() {
    $('selected').innerHTML = Object.entries(state.selected).map(([sid, item]) => `<div class="chosen-row"><strong>${esc(scenes.get(sid).name.split('_clipgt')[0])}</strong> · ${esc(item.clip_id)}<br>${esc(keys(item.categories).map(labelFor).join(' · '))}<br><span class="muted">${esc(item.reason)}</span></div>`).join('') || '<p class="muted">No frames chosen yet.</p>';
  }

  function openScene(sid) {
    currentScene = sid;
    const s = scenes.get(sid);
    $('scene-title').textContent = s.name;
    $('scene-status').textContent = `${s.status}. ${s.skipped_frames} extracted records lack CoT, plan or camera image. ${D.timing_note}`;
    const video = $('video');
    video.pause();
    if (s.video) {
      video.src = s.video;
      video.hidden = false;
    } else {
      video.removeAttribute('src');
      video.load();
      video.hidden = true;
    }
    $('frame-work').hidden = !s.frames.length;
    $('frame-image').hidden = !s.frames.length;
    const chosen = state.selected[sid];
    const index = chosen ? s.frames.findIndex(f => f.clip_id === chosen.clip_id) : 0;
    frameIndex = Math.max(0, index);
    if (s.frames.length) openFrame(frameIndex);
    renderLists();
  }

  function seek(time) {
    const v = $('video');
    if (!v.getAttribute('src')) return;
    const apply = () => {
      if (time >= 0 && (!Number.isFinite(v.duration) || time <= v.duration)) v.currentTime = time;
      else notify('Decision time is outside video duration. Check video offset and frame interval.', true);
    };
    if (v.readyState >= 1) apply();
    else v.addEventListener('loadedmetadata', apply, {
      once: true
    });
  }

  function openFrame(index) {
    const s = scenes.get(currentScene);
    frameIndex = Math.max(0, Math.min(s.frames.length - 1, index));
    const f = s.frames[frameIndex];
    $('timeline').max = s.frames.length - 1;
    $('timeline').value = frameIndex;
    $('previous').disabled = frameIndex === 0;
    $('next').disabled = frameIndex === s.frames.length - 1;
    $('frame-image').src = f.image;
    $('frame-image').hidden = false;
    $('cot').textContent = f.chain_of_thought;
    $('frame-id').textContent = `${f.clip_id} · time_now_us=${f.time_now_us} · video ≈ ${f.video_time_s.toFixed(2)}s`;
    const saved = state.selected[currentScene],
      mine = saved?.clip_id === f.clip_id ? saved : null;
    const p = proposal(f),
      c = mine ? mine.categories : p.categories;
    $('proposal-source').textContent = mine ? 'Categories confirmed for this selection.' : `Category suggestion: ${p.source}. Confirm or edit before selection.`;
    for (const axis of ['longitudinal', 'lateral']) $(axis).value = c ? c[axis] : '';
    document.querySelectorAll('[data-causal]').forEach(input => input.checked = !!c && c.causal_factors.includes(input.dataset.causal));
    $('confirm-tags').checked = !!mine;
    $('reason').value = mine ? mine.reason : '';
    $('remove').disabled = !saved;
    $('choose').textContent = saved && saved.clip_id !== f.clip_id ? 'Replace the selected frame for this clip' : 'Select this frame for the clip';
    seek(f.video_time_s);
    renderImpact();
  }

  function formCategories() {
    return {
      longitudinal: $('longitudinal').value,
      lateral: $('lateral').value,
      causal_factors: [...document.querySelectorAll('[data-causal]:checked')].map(i => i.dataset.causal).sort()
    };
  }

  function renderImpact() {
    const c = formCategories(),
      n = counts(currentScene);
    if (!c.longitudinal || !c.lateral) {
      $('impact').textContent = 'Classify both decision channels. “No decision” is different from unclassified.';
      return;
    }
    const changes = keys(c).map(k => `${labelFor(k)}: ${n[k]||0} → ${(n[k]||0)+1} (target ${state.targets[k]})`);
    $('impact').textContent = changes.join(' · ');
  }

  function choose() {
    try {
      if (!state.selector.trim()) throw Error('Enter your name or agent ID before selection.');
      if (!$('confirm-tags').checked) throw Error('Check the category confirmation box.');
      const categories = validCategories(formCategories()),
        reason = $('reason').value.trim();
      if (!reason) throw Error('Record why this frame should be included.');
      const f = scenes.get(currentScene).frames[frameIndex],
        previous = state.selected[currentScene] || null;
      const chosen = {
        clip_id: f.clip_id,
        categories: clone(categories),
        reason,
        confirmed: true,
        selector: state.selector,
        at: new Date().toISOString(),
        suggestion: clone(proposal(f))
      };
      state.selected[currentScene] = chosen;
      event(previous ? 'replace_selection' : 'select', {
        scene_id: currentScene,
        previous,
        selected: chosen
      });
      renderDashboard();
      renderLists();
      openFrame(frameIndex);
      notify('Saved one frame for this clip.');
    } catch (e) {
      notify(e.message, true);
    }
  }

  function download(name, payload) {
    const a = document.createElement('a'),
      url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {
        type: 'application/json'
      }));
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportExamples() {
    if (!Object.keys(state.selected).length) {
      notify('Select at least one frame first.', true);
      return;
    }
    const entries = Object.entries(state.selected).map(([sid, item]) => {
      const {
        frame: f,
        scene: s
      } = frames.get(item.clip_id);
      return {
        clip_id: f.clip_id,
        scene_id: sid,
        chain_of_thought: f.chain_of_thought,
        source_scene_file: f.source_scene_file,
        rollout_mp4: s.video_path,
        labels: {
          cot_decision_label: {
            source: 'frame_selector_confirmed_categories',
            ground_truth: false,
            high_level_decision: {
              longitudinal: item.categories.longitudinal,
              lateral: item.categories.lateral
            },
            critical_component_categories: item.categories.causal_factors
          }
        },
        selection: {
          ...clone(item),
          method: 'manual_category_guided',
          dataset_id: D.dataset_id,
          benchmark_sha256: D.benchmark_sha256,
          video_time_s: f.video_time_s,
          time_now_us: f.time_now_us
        }
      };
    });
    download('selected_examples.json', {
      schema_version: 1,
      dataset_id: D.dataset_id,
      generated_at: new Date().toISOString(),
      targets: state.targets,
      entries,
      selection_events: state.events
    });
    notify('Exported unanswered examples. Run the annotation builder with this file.');
  }

  function importSuggestions(raw) {
    if (raw.schema_version !== 1 || !Array.isArray(raw.labels)) throw Error('Expected schema_version=1 and labels array');
    const cotIds = new Set([...frames.values()].map(x => x.frame.cot_id)),
      seen = new Set(),
      updates = {};
    for (const row of raw.labels) {
      if (!cotIds.has(row.cot_id) || seen.has(row.cot_id)) throw Error('Unknown or duplicate CoT ID: ' + row.cot_id);
      seen.add(row.cot_id);
      validCategories(row.categories);
      if (typeof row.source !== 'string' || !row.source.trim()) throw Error('Suggestion needs a source');
      updates[row.cot_id] = {
        categories: clone(row.categories),
        source: row.source
      };
    }
    Object.assign(state.suggestions, updates);
    event('import_suggestions', {
      cot_ids: [...seen]
    });
    renderDashboard();
    renderLists();
    if (scenes.get(currentScene).frames.length) openFrame(frameIndex);
    notify(`Imported ${seen.size} provisional CoT labels. Confirmed selections retain their labels.`);
  }
  for (const axis of ['longitudinal', 'lateral']) $(axis).innerHTML = '<option value="">Unclassified — choose a category</option>' + Object.entries(D.taxonomy[axis]).map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('');
  $('causal').innerHTML = Object.entries(D.taxonomy.causal_factors).map(([v, l]) => `<label><input type="checkbox" data-causal="${v}">${esc(l)}</label>`).join('');
  $('focus').innerHTML += [...allKeys].map(k => `<option value="${k}">${k.split(':')[0]} · ${esc(labelFor(k))}</option>`).join('');
  $('selector-name').value = state.selector;
  $('selector-name').onchange = () => {
    state.selector = $('selector-name').value.trim();
    save();
  };
  $('focus').onchange = () => {
    focus = $('focus').value;
    renderLists();
  };
  $('clear-focus').onclick = () => {
    focus = '';
    $('focus').value = '';
    renderLists();
  };
  $('search').oninput = () => {
    search = $('search').value.trim().toLowerCase();
    renderLists();
  };
  $('only-selected').onchange = renderLists;
  $('timeline').oninput = () => openFrame(Number($('timeline').value));
  $('previous').onclick = () => openFrame(frameIndex - 1);
  $('next').onclick = () => openFrame(frameIndex + 1);
  $('from-video').onclick = () => {
    const fs = scenes.get(currentScene).frames;
    openFrame(fs.reduce((best, f, i) => Math.abs(f.video_time_s - $('video').currentTime) < Math.abs(fs[best].video_time_s - $('video').currentTime) ? i : best, 0));
  };
  for (const input of [...document.querySelectorAll('[data-causal]'), $('longitudinal'), $('lateral')]) input.onchange = () => {
    $('confirm-tags').checked = false;
    renderImpact();
  };
  $('choose').onclick = choose;
  $('remove').onclick = () => {
    const previous = state.selected[currentScene];
    if (!previous) return;
    delete state.selected[currentScene];
    event('remove_selection', {
      scene_id: currentScene,
      previous
    });
    renderDashboard();
    renderLists();
    openFrame(frameIndex);
  };
  $('save').onclick = () => download('frame_selection_session.json', state);
  $('export').onclick = exportExamples;
  $('import').onclick = () => $('import-file').click();
  $('import-labels').onclick = () => $('labels-file').click();
  $('import-file').onchange = async e => {
    try {
      const file = e.target.files[0];
      if (!file) return;
      const next = validateState(JSON.parse(await file.text()));
      state = next;
      $('selector-name').value = state.selector;
      save();
      renderDashboard();
      openScene(currentScene);
      notify('Session restored.');
    } catch (err) {
      notify('Import rejected: ' + err.message, true);
    } finally {
      e.target.value = '';
    }
  };
  $('labels-file').onchange = async e => {
    try {
      const file = e.target.files[0];
      if (file) importSuggestions(JSON.parse(await file.text()));
    } catch (err) {
      notify('Label import rejected: ' + err.message, true);
    } finally {
      e.target.value = '';
    }
  };
  $('video').addEventListener('error', () => notify('Video could not be loaded. Serve the workspace over HTTP and verify its media symlinks.', true));
  $('frame-image').addEventListener('error', () => notify('Camera frame could not be loaded. Verify the source extraction and media symlinks.', true));
  renderDashboard();
  openScene(currentScene);
})();