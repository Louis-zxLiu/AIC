"""Bind the archived shared trainer to RF, HBS and optional RGB lowlight."""
import sys,yaml
def main():
 cfg=yaml.safe_load(open(sys.argv[sys.argv.index('--config')+1],encoding='utf-8'))
 from aic.tools import train
 if cfg.get('lowlight_augmentation',{}).get('enabled'):
  import lowlight_hooks
  lowlight_hooks.install(cfg,train)
 import suite_runtime
 suite_runtime.install(cfg,train)
 from aic.rf_nas import entry
 entry.main()
if __name__=='__main__':main()
