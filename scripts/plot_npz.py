# scripts/plot_npz.py
import numpy as np, matplotlib.pyplot as plt, sys

path = sys.argv[1]
z = np.load(path)
pos = z["pos"]
gt  = z["gt"]
pr  = z["pred"]
err = z["err"]

fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)

# 速度ベクトルが多いので可視性向上のため間引き（任意）
step = max(1, int(pos.shape[0]//2000))  # ~2000本目安
ii = np.arange(0, pos.shape[0], step)

ax[0].quiver(pos[ii,0],pos[ii,1],gt[ii,0],gt[ii,1],angles='xy',scale_units='xy',scale=None,linewidths=0.3)
ax[0].set_title("Ground Truth velocity")

ax[1].quiver(pos[ii,0],pos[ii,1],pr[ii,0],pr[ii,1],angles='xy',scale_units='xy',scale=None,linewidths=0.3)
ax[1].set_title("Predicted velocity")

im=ax[2].scatter(pos[:,0],pos[:,1],c=err,s=3)
ax[2].set_title("Error |u_pred - u_gt|"); fig.colorbar(im,ax=ax[2])

for a in ax:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])

out_png = path.replace(".npz",".png")
plt.savefig(out_png,dpi=180)
print("Saved:", out_png)
