/* Quietly: one command dispatcher for buttons, keyboard and real sEMG events. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const WORDS = ['open', 'close', 'next', 'music', 'start', 'stop', 'previous', 'quieter', 'louder', 'photo'];
  const state = {view: 'home', photo: 0, music: 0, photos: [], tracks: [], volume: 50, threshold: 0};
  const audio = $('audio');
  const lightbox = $('lightbox');
  const logs = [];
  const seen = new Set();
  let queue = Promise.resolve(), toastTimer, eventStream, lastSignal = 0;
  const number = value => String(value + 1).padStart(2, '0');
  const icon = name => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const filename = item => item?.name || '';
  const label = item => filename(item).replace(/\.[^.]+$/, '');
  const trackInfo = item => {
    const parts = label(item).split(' - ');
    return parts.length > 1 ? {artist: parts.shift(), title: parts.join(' - ')} : {artist: 'Local music', title: label(item)};
  };
  const time = seconds => Number.isFinite(seconds) ? `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}` : '--:--';

  function toast(message) {
    $('toast').textContent = message;
    $('toast').hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4000);
  }
  function setView(view) {
    if (lightbox.open) lightbox.close();
    if (view !== 'music') audio.pause();
    state.view = view;
    ['home', 'photo', 'music'].forEach(name => { $(`${name}-view`).hidden = name !== view; });
    const commands = view === 'home' ? ['photo', 'music'] : view === 'photo' ? ['previous', 'next', 'open', 'close', 'music'] : ['previous', 'next', 'start', 'stop', 'quieter', 'louder', 'photo'];
    $('context-commands').replaceChildren(...commands.map(word => commandButton(word)));
    window.scrollTo({top: 0, behavior: 'instant'});
  }
  function commandButton(word) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'command-chip';
    button.dataset.command = word;
    button.textContent = word;
    button.setAttribute('aria-label', `Test command ${word}`);
    return button;
  }
  function updatePhoto() {
    const item = state.photos[state.photo];
    const count = state.photos.length;
    $('photo-count').textContent = `${count} photos in your collection`;
    $('photo-number').replaceChildren(document.createTextNode(count ? number(state.photo) : '00'));
    const total = document.createElement('span'); total.textContent = `/ ${String(count).padStart(2, '0')}`;
    $('photo-number').append(total);
    $('photo-title').textContent = item ? label(item) : 'No photos yet';
    $('photo-empty').hidden = !!item;
    $('photo-preview-button').hidden = !item;
    if (item) {
      for (const id of ['photo-preview', 'lightbox-image']) {
        $(id).src = item.url;
        $(id).alt = label(item);
      }
      $('lightbox-title').textContent = label(item);
      $('lightbox-counter').textContent = `${number(state.photo)} / ${String(count).padStart(2, '0')}`;
    }
    [...$('photo-strip').children].forEach((el, index) => {
      el.classList.toggle('selected', index === state.photo);
      el.setAttribute('aria-pressed', String(index === state.photo));
    });
    $('photo-strip').children[state.photo]?.scrollIntoView({block: 'nearest', inline: 'nearest', behavior: 'smooth'});
  }
  function updatePlaying() {
    const playing = !audio.paused && !audio.ended;
    $('play-button').innerHTML = icon(playing ? 'pause' : 'play');
    $('play-button').classList.toggle('is-playing', playing);
    $('play-button').setAttribute('aria-label', playing ? 'stop — Pause' : 'start — Play');
    $('player-vinyl').classList.toggle('playing', playing);
    $('record-status').innerHTML = `<i></i> ${playing ? 'NOW PLAYING' : audio.currentTime > 0 ? 'ON PAUSE' : 'READY TO PLAY'}`;
    [...$('track-list').children].forEach((row, index) => {
      row.classList.toggle('selected', index === state.music);
      row.setAttribute('aria-pressed', String(index === state.music));
      row.querySelector('.track-indicator').innerHTML = index === state.music ? icon(playing ? 'volume' : 'music') : '';
    });
  }
  function updateProgress() {
    $('elapsed').textContent = time(audio.currentTime);
    $('duration').textContent = time(audio.duration);
    $('seek').value = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.currentTime / audio.duration * 1000 : 0;
    $('seek').disabled = !Number.isFinite(audio.duration) || audio.duration <= 0;
  }
  async function play() {
    if (!state.tracks.length) return 'Music library is empty';
    try { await audio.play(); return 'Now playing'; }
    catch (error) {
      const message = error.name === 'NotAllowedError' ? 'Click Play once to enable browser audio.' : 'Unable to play. Check the file and browser format support.';
      toast(message); return message;
    }
  }
  async function selectTrack(index, continuePlaying = false) {
    audio.pause();
    state.music = index;
    const item = state.tracks[index];
    const info = item ? trackInfo(item) : {artist: 'Add audio to the music folder', title: 'Waiting for your first track'};
    $('track-title').textContent = info.title;
    $('track-artist').textContent = info.artist;
    $('record-artist').textContent = info.artist.toUpperCase();
    $('track-counter').textContent = `${item ? number(index) : '00'} / ${String(state.tracks.length).padStart(2, '0')}`;
    $('music-empty').hidden = !!item;
    $('play-button').disabled = !item;
    if (item) { audio.src = item.url; audio.load(); }
    else { audio.removeAttribute('src'); audio.load(); }
    updatePlaying(); updateProgress();
    if (continuePlaying && item) return play();
    return item ? `Selected: ${info.title}` : 'Music library is empty';
  }
  function volume(value) {
    state.volume = Math.max(0, Math.min(100, Math.round(value)));
    audio.volume = state.volume / 100;
    $('volume').value = state.volume;
    $('volume-value').textContent = `${state.volume}%`;
    return `Volume ${state.volume}%`;
  }
  function feedback(word, result, input) {
    const manual = input.source === 'manual' || input.source === 'keyboard';
    $('feedback-label').textContent = manual ? (input.source === 'keyboard' ? 'KEYBOARD TEST' : 'MANUAL TEST') : 'sEMG INPUT';
    $('last-command').textContent = word;
    $('command-result').textContent = result;
    $('confidence').textContent = manual ? 'Test input · Not a model prediction' : Number.isFinite(input.confidence) ? `Model confidence ${(input.confidence * 100).toFixed(1)}%` : 'No confidence provided';
    const dock = document.querySelector('.command-dock');
    dock.classList.remove('flash'); void dock.offsetWidth; dock.classList.add('flash');
    document.querySelectorAll('.command-chip').forEach(el => el.classList.toggle('active', el.dataset.command === word));
    logs.unshift({word, result, source: manual ? 'Test' : 'sEMG', time: new Date().toLocaleTimeString('en-GB', {hour12: false})});
    logs.splice(20);
    $('event-log').replaceChildren(...logs.map(log => {
      const li = document.createElement('li'), t = document.createElement('time'), b = document.createElement('b'), s = document.createElement('span');
      t.textContent = log.time; b.textContent = log.word; s.textContent = `${log.source} · ${log.result}`;
      li.append(t, b, s); return li;
    }));
  }
  async function dispatch(input) {
    if (typeof input === 'string') input = {word: input};
    if (input?.board && Date.now() - input.sent_at >= 1000) return {ok: false, result: 'Expired board command ignored'};
    if (!input || typeof input.word !== 'string') return {ok: false, result: 'Missing command'};
    const word = input.word.trim().toLowerCase();
    if (!WORDS.includes(word)) { toast(`Unknown command: ${word.slice(0, 30)}`); return {ok: false, result: 'Unknown command'}; }
    if (input.id) {
      if (seen.has(input.id)) return {ok: false, result: 'Duplicate event ignored'};
      seen.add(input.id); if (seen.size > 500) seen.delete(seen.values().next().value);
    }
    const manual = ['manual', 'keyboard'].includes(input.source);
    state.threshold = Math.max(0, Math.min(100, Number($('threshold').value) || 0));
    const validConfidence = typeof input.confidence === 'number' && Number.isFinite(input.confidence) && input.confidence >= 0 && input.confidence <= 1;
    if (!manual && ((input.confidence != null && !validConfidence) || (state.threshold > 0 && (!validConfidence || input.confidence * 100 < state.threshold)))) {
      const result = validConfidence ? 'Below confidence threshold. Not executed.' : 'Missing or invalid confidence. Not executed.';
      feedback(word, result, input); return {ok: false, result};
    }
    let result;
    if (word === 'music' || word === 'photo') { setView(word); result = word === 'music' ? 'Music library ready' : 'Photo library ready'; }
    else if (word === 'next' || word === 'previous') {
      const step = word === 'next' ? 1 : -1;
      if (word === 'previous' && state.view !== 'home' && state[state.view] === 0) {
        setView('home');
        result = 'Back home. Choose music or photo.';
      } else if (state.view === 'photo' && state.photos.length) {
        state.photo = (state.photo + step + state.photos.length) % state.photos.length;
        updatePhoto(); result = `Photo ${number(state.photo)} / ${String(state.photos.length).padStart(2, '0')}`;
      } else if (state.view === 'music' && state.tracks.length) {
        const wasPlaying = !audio.paused && !audio.ended;
        result = await selectTrack((state.music + step + state.tracks.length) % state.tracks.length, wasPlaying);
      } else result = state.view === 'home' ? 'Choose a space with music or photo first.' : 'This library is empty';
    } else if (word === 'open' || word === 'close') {
      if (state.view !== 'photo') result = 'Use open / close in the photo library.';
      else if (!state.photos.length) result = 'Photo library is empty';
      else if (word === 'open') {
        if ($('settings').open) $('settings').close();
        if (!lightbox.open) lightbox.showModal();
        result = 'Full-size photo opened';
      } else { if (lightbox.open) lightbox.close(); result = 'Full-size photo closed'; }
    } else if (state.view !== 'music') result = 'Use music to enter the music library first.';
    else if (word === 'start') result = await play();
    else if (word === 'stop') { audio.pause(); result = 'Playback paused'; }
    else if (word === 'louder') result = volume(state.volume + 10);
    else if (word === 'quieter') result = volume(state.volume - 10);
    feedback(word, result, input);
    return {ok: true, result};
  }
  function send(input) {
    queue = queue.then(() => dispatch(input)).catch(error => {
      console.error(error); toast('Command failed. Please try again.'); return {ok: false, result: error.message};
    });
    return queue;
  }

  function buildLibrary(manifest) {
    state.photos = Array.isArray(manifest.photos) ? manifest.photos : [];
    state.tracks = Array.isArray(manifest.music) ? manifest.music : [];
    const hero = state.photos[1] || state.photos[0];
    if (hero) $('home-photo').style.backgroundImage = `url(${JSON.stringify(hero.url)})`;
    $('home-photo-count').textContent = `${String(state.photos.length).padStart(2, '0')} PHOTOS`;
    $('home-music-count').textContent = `${String(state.tracks.length).padStart(2, '0')} TRACKS`;
    $('music-count').textContent = `${state.tracks.length} tracks in your collection`;
    $('queue-count').textContent = `${state.tracks.length} TRACKS`;
    $('photo-strip').replaceChildren(...state.photos.map((item, index) => {
      const button = document.createElement('button'), img = document.createElement('img'), span = document.createElement('span');
      button.className = 'thumbnail'; button.type = 'button'; button.setAttribute('aria-label', `Select photo ${label(item)}`);
      img.src = item.url; img.alt = label(item); img.loading = 'lazy'; span.textContent = number(index);
      button.append(img, span);
      button.addEventListener('click', () => { state.photo = index; updatePhoto(); }); return button;
    }));
    $('track-list').replaceChildren(...state.tracks.map((item, index) => {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'track-row';
      const info = trackInfo(item);
      button.innerHTML = '<span class="track-index"></span><span class="track-row-copy"><strong></strong><small></small></span><span class="track-indicator"></span>';
      button.querySelector('.track-index').textContent = number(index);
      button.querySelector('strong').textContent = info.title;
      button.querySelector('small').textContent = info.artist;
      button.setAttribute('aria-label', `Select track ${info.title}`);
      button.addEventListener('click', () => { queue = queue.then(() => selectTrack(index, !audio.paused && !audio.ended)); });
      return button;
    }));
    updatePhoto(); selectTrack(0); volume(50); setView('home');
  }
  function drawSignal(data) {
    const channels = data.channels;
    if (!Array.isArray(channels) || channels.length < 1) return;
    const canvas = $('signal-canvas'), ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    channels.slice(0, 2).forEach((channel, index) => {
      const values = channel.slice(-500).filter(Number.isFinite);
      if (values.length < 2) return;
      const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
      ctx.strokeStyle = index ? '#9b854f' : '#416a52'; ctx.lineWidth = 2; ctx.beginPath();
      values.forEach((v, i) => {
        const x = i / (values.length - 1) * canvas.width, y = index * 80 + 68 - ((v - lo) / span) * 54;
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }); ctx.stroke();
    });
    lastSignal = Date.now();
    $('signal-status').textContent = `Receiving · ${data.sample_rate ? `${data.sample_rate} Hz` : 'Auto scale'}`;
  }
  function connect() {
    if (location.protocol === 'file:') {
      $('connection').querySelector('span').textContent = 'Offline preview · Manual input';
      $('bridge-info').textContent = 'Offline file preview. Run start.cmd to enable the live sEMG bridge and refresh your media library.';
      return;
    }
    $('bridge-info').textContent = `Local endpoint: POST ${location.origin}/api/command`;
    eventStream = new EventSource('/api/events');
    eventStream.addEventListener('open', () => {
      $('connection').classList.add('online');
      $('connection').querySelector('span').textContent = 'Command bridge connected';
    });
    eventStream.addEventListener('error', () => {
      $('connection').classList.remove('online');
      $('connection').querySelector('span').textContent = 'Bridge offline · Reconnecting';
    });
    eventStream.addEventListener('command', event => { try { const data = JSON.parse(event.data); if (!data.board || Date.now() - data.sent_at < 1000) send({...data, source: 'semg'}); } catch { toast('Invalid command data received.'); } });
    for (const kind of ['board', 'envelope']) eventStream.addEventListener(kind, event => { try { window.dispatchEvent(new CustomEvent('quietly-' + kind, {detail: JSON.parse(event.data)})); } catch {} });
    eventStream.addEventListener('signal', event => { try { drawSignal(JSON.parse(event.data)); } catch { /* Invalid waveform never becomes a command. */ } });
  }

  document.addEventListener('click', event => {
    if (event.target.closest('[data-home]')) {
      setView('home'); $('command-result').textContent = 'Mouth music or photo to enter';
      return;
    }
    const button = event.target.closest('[data-command]');
    if (button) send({word: button.dataset.command, source: 'manual'});
  });
  $('home-button').addEventListener('click', () => { setView('home'); $('command-result').textContent = 'Mouth music or photo to enter'; });
  $('photo-preview-button').addEventListener('click', () => send({word: 'open', source: 'manual'}));
  $('close-lightbox').addEventListener('click', () => send({word: 'close', source: 'manual'}));
  lightbox.addEventListener('cancel', event => { event.preventDefault(); send({word: 'close', source: 'keyboard'}); });
  $('play-button').addEventListener('click', () => send({word: audio.paused || audio.ended ? 'start' : 'stop', source: 'manual'}));
  $('volume').addEventListener('input', event => volume(Number(event.target.value)));
  $('seek').addEventListener('input', event => { if (Number.isFinite(audio.duration)) audio.currentTime = Number(event.target.value) / 1000 * audio.duration; });
  ['play', 'pause', 'ended'].forEach(name => audio.addEventListener(name, updatePlaying));
  ['timeupdate', 'loadedmetadata', 'durationchange', 'emptied'].forEach(name => audio.addEventListener(name, updateProgress));
  audio.addEventListener('ended', () => { $('command-result').textContent = 'Track finished. Use start to replay or next to browse.'; });
  audio.addEventListener('error', () => { if (audio.getAttribute('src')) toast('Unable to load this track. Check the file and format.'); });
  $('photo-preview').addEventListener('error', () => { toast('Unable to load this photo. Check the file and refresh.'); });
  $('settings-button').addEventListener('click', () => $('settings').showModal());
  $('close-settings').addEventListener('click', () => $('settings').close());
  $('settings').addEventListener('click', event => { if (event.target === $('settings')) { const r = event.target.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.target.close(); } });
  $('test-commands').append(...WORDS.map(commandButton));
  $('command-form').addEventListener('submit', event => { event.preventDefault(); send({word: $('command-input').value, source: 'manual'}); $('command-input').value = ''; });
  $('threshold').addEventListener('input', event => { state.threshold = Math.max(0, Math.min(100, Number(event.target.value) || 0)); });
  $('threshold').addEventListener('change', event => { event.target.value = state.threshold; });
  $('fullscreen-button').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch { toast('Fullscreen is unavailable. Try your browser F11 shortcut.'); }
  });
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.repeat || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable || $('settings').open) return;
    if ((event.key === 'Enter' || event.key === ' ') && event.target.closest('button')) return;
    const key = event.key.toLowerCase();
    const map = {m: 'music', p: 'photo', arrowleft: 'previous', arrowright: 'next', arrowup: 'louder', arrowdown: 'quieter', enter: 'open', ' ': audio.paused || audio.ended ? 'start' : 'stop'};
    if (map[key]) { event.preventDefault(); send({word: map[key], source: 'keyboard'}); }
    else if (key === 'h') { event.preventDefault(); setView('home'); }
  });
  window.addEventListener('semg-command', event => send({...event.detail, source: 'semg'}));
  window.Quietly = Object.freeze({send: input => send(typeof input === 'string' ? {word: input, source: 'semg'} : {...input, source: 'semg'}), getState: () => ({view: state.view, photoIndex: state.photo, musicIndex: state.music, volume: state.volume, playing: !audio.paused, largePhoto: lightbox.open, photoCount: state.photos.length, musicCount: state.tracks.length})});
  async function init() {
    let manifest = window.MEDIA_MANIFEST || {photos: [], music: []};
    if (location.protocol !== 'file:') {
      try { const response = await fetch('/api/media'); if (response.ok) manifest = await response.json(); }
      catch { toast('Unable to refresh media. Using the saved library.'); }
    }
    buildLibrary(manifest); connect();
  }
  setInterval(() => { if (lastSignal && Date.now() - lastSignal > 3000) $('signal-status').textContent = 'Signal paused · Last frame retained'; }, 1000);
  init();
})();

