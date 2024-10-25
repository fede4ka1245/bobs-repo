import { styled } from '@mui/material/styles';
import MuiDrawer from '@mui/material/Drawer';

export default styled(MuiDrawer)(({ theme }) => ({
  '& .MuiDrawer-paper': {
    height: 'calc(100vh - 80px)',
    width: '100vw',
    background: 'var(--bg-color)',
    overflow: 'visible',
    borderRadius: 'var(--border-radius-lg) var(--border-radius-lg) 0 0',
    margin: 'auto 0',
    maxWidth: 'var(--max-content)',
  },

  '& .MuiBackdrop-root': {
    background: 'rgba(128, 128, 128, 0.6)',
  },
}));
