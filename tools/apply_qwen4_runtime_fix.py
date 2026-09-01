#!/usr/bin/env python3
"""Apply the complete reviewed Qwen4 runtime semantic/perf patch."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tools" / "apply_qwen4_sigmoid_fix.py"), run_name="__main__")

path = ROOT / "logan-qwen4" / "src" / "lib.rs"
text = path.read_text()
old = """        if self.gdn_metal[li].is_none() {
            self.gdn_metal[li] = Self::build_gdn_metal(layer, &self.cfg);
        }
"""
new = """        if self.gdn_metal[li].is_none() {
            let built = Self::build_gdn_metal(layer, &self.cfg);
            if let Some(gm) = built.as_ref() {
                // A RAM snapshot can be restored before the aligned buffers
                // exist. Seed the newly-created authoritative buffers from the
                // CPU state exactly once so lazy initialization never erases a
                // restored recurrent/conv state. Fresh models simply copy zero.
                let state_len = vheads * kd * vd;
                let conv_len = cdim * (kk - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.gdn_s[li].as_ptr(),
                        gm.state,
                        state_len,
                    );
                    std::ptr::copy_nonoverlapping(
                        self.gdn_conv[li].as_ptr(),
                        gm.conv_state,
                        conv_len,
                    );
                }
            }
            self.gdn_metal[li] = built;
        }
"""
count = text.count(old)
if count != 1:
    raise SystemExit(f"aligned GDN init: expected one source match, found {count}")
path.write_text(text.replace(old, new, 1))
print(f"patched {path.relative_to(ROOT)} aligned-state initialization")
