import pyqet

dm_tiles = pyqet.upb.load_upb('tiles', return_bes=True)[1]
dm_pyramid = pyqet.upb.load_upb('pyramid', return_bes=True)[1]

fig,ax,all_data = pyqet.bes.plot_dm0_dm1_plane(dm0=dm_tiles, dm1=dm_pyramid, dimA=3, dimB=3, num_eig0=0, num_point=201, pureb_kext=[8,32], tag_cha=True, label0='Tiles', label1='Pyramid')
ax.legend(fontsize='large', ncols=2, loc='lower right')
fig.tight_layout()