/* SPDX-License-Identifier: Apache-2.0 */
'use strict';
(() => {
  const steps = [{
    target: '#dashboard > div:first-child',
    title: 'Start with sample-size gaps',
    paragraphs: [
      'These tables count cases for longitudinal decisions, lateral decisions, and causal factors. Existing (blue) is the benchmark; Added (green) is your confirmed selections.',
      'Set a Target; Gap is what remains to reach it. Avail. counts unselected clips with matching provisional tags. Look for small Existing counts and remaining gaps when deciding what to add.'
    ]
  }, {
    target: '#category-focus',
    title: 'Focus on a category to expand',
    paragraphs: [
      'Choose a focus category here, or click its name in a table. The clip and candidate-decision lists narrow to matching suggestions.',
      'Clear focus to see unclassified candidates too. Selected clips only lets you review your current choices.'
    ]
  }, {
    target: '#scene-controls',
    title: 'Choose a source clip',
    paragraphs: [
      'Search by clip or CoT text, then open a clip from the list below. Clips are ordered by how much their suggested categories could reduce the current shortfalls.',
      'Check the preparation status and decision count. A clip needs extracted camera input, CoT, and plan before a frame can be selected. You make the final choice.'
    ]
  }, {
    target: '#scene-status',
    when: () => document.getElementById('frame-work').hidden,
    title: 'This clip has no selectable decisions',
    paragraphs: [
      'The status explains whether this clip needs frame extraction or is excluded. A video timestamp alone is not a complete benchmark example.',
      'Close the tour, open a clip with usable extracted decisions, and restart it to see the frame-selection steps. For now, the tour skips unavailable controls.'
    ]
  }, {
    target: '#video',
    title: 'Browse the video for context',
    paragraphs: [
      'Play or scrub the rollout to find a useful moment. Then use “Use nearest decision to video time” to open an extracted decision near that moment.',
      'Later footage includes replanning; it does not necessarily show execution of the plan at the selected decision. Use the exact input frame and CoT to decide what to include.'
    ]
  }, {
    target: '#frame-image',
    title: 'Check the exact model input',
    paragraphs: [
      'This is the camera image actually paired with the current decision. The video seek time is approximate, so verify the scene here.',
      'Choose a usable, interpretable example of the category you want to expand. There can be at most one selected frame per source clip.'
    ]
  }, {
    target: '#decision-navigation',
    title: 'Choose the decision timestamp',
    paragraphs: [
      'Use Previous decision, Next decision, or the slider to browse extracted decisions. “Use nearest decision to video time” connects your video browsing to these available frames.',
      'The Candidate decisions list below offers another way to jump to a CoT. Browsing does not commit a selection.'
    ]
  }, {
    target: '#cot',
    title: 'Read what the CoT actually states',
    paragraphs: [
      'This text belongs to the current decision frame. Use its stated intent and causal factors to categorize the example.',
      'The suggestion source appears below it. Suggestions are provisional: a matching tag does not establish whether the CoT is reliable or the plan is safe.'
    ]
  }, {
    target: '#category-editor',
    title: 'Verify the categories',
    paragraphs: [
      'Choose one longitudinal and one lateral category. Use the absent-decision option when no decision is stated in that channel. Several causal factors can apply.',
      'Review or correct every suggestion, then tick the confirmation box. Changing a category clears that confirmation. Imported CoT suggestions need the same review.'
    ]
  }, {
    target: '#selection-reason',
    title: 'Record why this frame belongs',
    paragraphs: [
      'Write a concrete inclusion reason, such as adding a rare speed-adaptation example from a new clip. Document any quality or interest criteria behind your choice.',
      'The impact preview shows how the current categories affect the remaining shortfall, accounting for a previous selection from this clip.'
    ]
  }, {
    target: '#selection-actions',
    title: 'Keep one frame per clip',
    paragraphs: [
      'After confirming the categories and entering a reason, select the frame. The dashboard and selected-example list update.',
      'Selecting another timestamp from the same clip replaces the earlier frame. It never adds a second example from that clip. Remove clip selection clears that clip’s contribution.'
    ]
  }, {
    target: '#selected-panel',
    title: 'Review the growing selection',
    paragraphs: [
      'Each entry shows the chosen frame, confirmed categories, and inclusion reason. Check this list and the updated counts as you expand the benchmark.',
      'Revisit a clip to change its chosen timestamp. Keep using the remaining gaps to guide which category you look for next.'
    ]
  }, {
    target: '#session-backup',
    title: 'Save a session backup',
    paragraphs: [
      'Enter your name or agent ID in the Selector field. Browser autosave keeps work in this workspace; Save selection session downloads a backup with selections, targets, and history.',
      'Use Restore session to resume that backup in the matching workspace. Save before moving the workspace or rebuilding its candidate data.'
    ]
  }, {
    target: '#export',
    title: 'Export for independent annotation',
    paragraphs: [
      'Export selected examples downloads the chosen frames, categories, and selection provenance with the evaluation answers still unanswered.',
      'Use that file with the existing annotation builder for the next stage. Finish this tour to start working; you can reopen it at any time.'
    ]
  }];

  const dialog = document.createElement('dialog');
  dialog.id = 'selector-tour';
  dialog.setAttribute('aria-labelledby', 'tour-title');
  dialog.setAttribute('aria-describedby', 'tour-description');
  dialog.innerHTML = `
    <div id="tour-highlight" aria-hidden="true"></div>
    <div id="tour-card">
      <div class="tour-topline"><span id="tour-count"></span><button id="tour-close" type="button">Close tour</button></div>
      <div aria-live="polite" aria-atomic="true"><h2 id="tour-title"></h2><div id="tour-description"></div></div>
      <p class="tour-hint">Selections stay unchanged. Esc closes the tour.</p>
      <div class="tour-actions"><button id="tour-back" type="button">Back</button><button id="tour-next" type="button" class="primary">Next</button></div>
    </div>`;
  document.body.append(dialog);
  const $ = id => document.getElementById(id);
  const card = $('tour-card');
  const highlight = $('tour-highlight');
  let activeSteps = [],
    index = 0,
    previousFocus, previousScroll, animation;

  function position(allowScroll = false) {
    if (!dialog.open) return;
    const target = document.querySelector(activeSteps[index].target);
    const edge = 14,
      gap = 18;
    const width = window.innerWidth,
      height = window.innerHeight;
    const cw = card.offsetWidth,
      ch = card.offsetHeight;
    let rect = target.getBoundingClientRect();
    const hasSideRoom = () => rect.right + gap + cw <= width - edge || rect.left - gap - cw >= edge;
    if (allowScroll && !hasSideRoom() && height - rect.bottom - gap - edge < ch && rect.top - gap - edge < ch) {
      // Make room below wide targets; the document boundary may leave room above instead.
      window.scrollBy({
        top: rect.top - edge - 5,
        behavior: 'instant'
      });
      rect = target.getBoundingClientRect();
    }
    const clamp = (value, max) => Math.max(edge, Math.min(value, max));
    let left = clamp(rect.left, width - cw - edge);
    let top, docked = false;
    if (rect.right + gap + cw <= width - edge) {
      left = rect.right + gap;
      top = clamp(rect.top, height - ch - edge);
    } else if (rect.left - gap - cw >= edge) {
      left = rect.left - gap - cw;
      top = clamp(rect.top, height - ch - edge);
    } else if (height - rect.bottom - gap - edge >= ch) {
      top = rect.bottom + gap;
    } else if (rect.top - gap - edge >= ch) {
      top = rect.top - gap - ch;
    } else {
      // On short screens, show the visible part of the target above a docked card.
      top = height - ch - edge;
      docked = true;
    }
    card.style.left = `${left}px`;
    card.style.top = `${Math.max(edge, top)}px`;
    const x = Math.max(4, rect.left - 5),
      y = Math.max(4, rect.top - 5);
    const right = Math.min(width - 4, rect.right + 5);
    const bottom = Math.min(height - 4, rect.bottom + 5, docked ? top - gap : height);
    highlight.hidden = right <= x || bottom <= y;
    Object.assign(highlight.style, {
      left: `${x}px`,
      top: `${y}px`,
      width: `${Math.max(0, right - x)}px`,
      height: `${Math.max(0, bottom - y)}px`
    });
  }

  function schedulePosition() {
    cancelAnimationFrame(animation);
    animation = requestAnimationFrame(() => position());
  }
  const observer = new ResizeObserver(schedulePosition);

  function showStep() {
    const step = activeSteps[index];
    $('tour-count').textContent = `Step ${index + 1} of ${activeSteps.length}`;
    $('tour-title').textContent = step.title;
    $('tour-description').replaceChildren(...step.paragraphs.map(text => {
      const p = document.createElement('p');
      p.textContent = text;
      return p;
    }));
    $('tour-back').disabled = index === 0;
    $('tour-next').textContent = index === activeSteps.length - 1 ? 'Finish' : 'Next';
    const target = document.querySelector(step.target);
    dialog.dataset.target = step.target;
    target.scrollIntoView({
      block: 'center',
      inline: 'nearest',
      behavior: 'instant'
    });
    position(true);
    observer.disconnect();
    observer.observe(target);
    observer.observe(card);
    $('tour-next').focus({
      preventScroll: true
    });
  }

  $('start-tour').addEventListener('click', () => {
    activeSteps = steps.filter(step => (!step.when || step.when()) && document.querySelector(step.target)?.getClientRects().length);
    previousFocus = document.activeElement;
    previousScroll = {
      left: window.scrollX,
      top: window.scrollY
    };
    index = 0;
    dialog.showModal();
    showStep();
  });
  $('tour-next').addEventListener('click', () => {
    if (index === activeSteps.length - 1) dialog.close();
    else {
      index++;
      showStep();
    }
  });
  $('tour-back').addEventListener('click', () => {
    if (index > 0) {
      index--;
      showStep();
    }
  });
  $('tour-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    observer.disconnect();
    cancelAnimationFrame(animation);
    previousFocus.focus({
      preventScroll: true
    });
    window.scrollTo({
      ...previousScroll,
      behavior: 'instant'
    });
  });
  window.addEventListener('resize', schedulePosition);
  window.addEventListener('scroll', schedulePosition, {
    capture: true,
    passive: true
  });
})();
