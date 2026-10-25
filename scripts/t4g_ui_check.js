// Verify the actual UI slider mappings without starting or modifying the user's app.
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const sliders = html.match(/const UI_SLIDERS = (\[[\s\S]*?\n\]);/)[1];
const defaults = html.match(/function defaultNaturalParams\(\) \{[\s\S]*?\n\}/)[0];
const context = {};
vm.createContext(context);
vm.runInContext(`this.sliders = ${sliders}; ${defaults}; this.defaults = defaultNaturalParams();`, context);
for (const key of ['midtone', 'cam_mix', 'chroma_preserve']) {
  const slider = context.sliders.find(s => s.key === key);
  const target = key === 'midtone' ? 0.23 : 0.40;
  if (slider.toUI(target) !== slider.def || slider.fromUI(slider.def) !== target || context.defaults[key] !== target) {
    throw new Error(`UI default mismatch: ${key}`);
  }
}
const mid = context.sliders.find(s => s.key === 'midtone');
if (Math.abs(mid.fromUI(-100) - 0.10) > 1e-9 || Math.abs(mid.fromUI(100) - 0.45) > 1e-9) {
  throw new Error('Midtone endpoints changed');
}
new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
console.log('UI script syntax, defaults and midtone endpoints passed');
