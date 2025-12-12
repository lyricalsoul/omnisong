# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ui_player.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('Assets.car', '.'), # this is so i can use an icon composer icon, which supports liquid glass
        ('omni.pth', '.'),
        ('sounds/', 'sounds/'),],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Omnisong',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='Omnisong',
)
app = BUNDLE(
    coll,
    name='Omnisong.app',
    icon=None,
    bundle_identifier=None,
    info_plist={"CFBundleIconName": "OmnisongIcon"}, # use .icon file from Assets.car
)
